"""Local AI generation against a fake ComfyUI HTTP server (no GPU, no models needed).

The fake validates workflows the way ComfyUI does (known node types, link targets and
output types, model names in the installed lists) and answers with ComfyUI's JSON
shapes, so these tests catch broken wiring before Kevin's GPU ever sees it.
"""
import email.parser
import email.policy
import io
import json
import socket
import threading
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw

from pixel_pipeline import pipeline, settings_for
from pixel_pipeline.generate import (
    DEFAULT_CHECKPOINT,
    DEFAULT_LCM_LORA,
    DEFAULT_LORA,
    ComfyClient,
    ComfyUIError,
    ComfyUIModelMissing,
    ComfyUIUnavailable,
    ComfyUIWorkflowError,
    Generated,
    build_workflow,
    check_setup,
    fill_template,
    generate_images,
    generate_sprite,
    generation_size,
    load_template,
    prepare_init_image,
    sprite_prompt,
    template_placeholders,
)

TEMPLATES = Path(__file__).resolve().parents[1] / "tools" / "comfyui"

# --------------------------------------------------------------------------- fake ComfyUI

OUTPUTS = {"CheckpointLoaderSimple": ["MODEL", "CLIP", "VAE"], "LoraLoader": ["MODEL", "CLIP"],
           "ModelSamplingDiscrete": ["MODEL"], "VAELoader": ["VAE"], "CLIPTextEncode": ["CONDITIONING"],
           "EmptyLatentImage": ["LATENT"], "LoadImage": ["IMAGE", "MASK"], "VAEEncode": ["LATENT"],
           "KSampler": ["LATENT"], "VAEDecode": ["IMAGE"], "SaveImage": [], "PreviewImage": [],
           # Hunyuan3D 2 (comfy_extras/nodes_hunyuan3d.py, nodes_save_3d.py)
           "ImageOnlyCheckpointLoader": ["MODEL", "CLIP_VISION", "VAE"],
           "CLIPVisionEncode": ["CLIP_VISION_OUTPUT"],
           "Hunyuan3Dv2Conditioning": ["CONDITIONING", "CONDITIONING"],
           "Hunyuan3Dv2ConditioningMultiView": ["CONDITIONING", "CONDITIONING"],
           "EmptyLatentHunyuan3Dv2": ["LATENT"], "ModelSamplingAuraFlow": ["MODEL"],
           "VAEDecodeHunyuan3D": ["VOXEL"], "VoxelToMesh": ["MESH"], "SaveGLB": []}
SAMPLERS = ["euler", "euler_ancestral", "dpmpp_2m", "lcm"]
SCHEDULERS = ["normal", "karras", "sgm_uniform", "simple"]
INPUTS = {
    "CheckpointLoaderSimple": {"ckpt_name": "checkpoints"},
    "LoraLoader": {"model": "MODEL", "clip": "CLIP", "lora_name": "loras", "strength_model": float,
                   "strength_clip": float},
    "ModelSamplingDiscrete": {"model": "MODEL", "sampling": ["eps", "v_prediction", "lcm", "x0"],
                              "zsnr": bool},
    "VAELoader": {"vae_name": "vae"},
    "CLIPTextEncode": {"text": str, "clip": "CLIP"},
    "EmptyLatentImage": {"width": int, "height": int, "batch_size": int},
    "LoadImage": {"image": "uploads"},
    "VAEEncode": {"pixels": "IMAGE", "vae": "VAE"},
    "KSampler": {"model": "MODEL", "seed": int, "steps": int, "cfg": float, "sampler_name": SAMPLERS,
                 "scheduler": SCHEDULERS, "positive": "CONDITIONING", "negative": "CONDITIONING",
                 "latent_image": "LATENT", "denoise": float},
    "VAEDecode": {"samples": "LATENT", "vae": "VAE"},
    "SaveImage": {"images": "IMAGE", "filename_prefix": str},
    "PreviewImage": {"images": "IMAGE"},
    "ImageOnlyCheckpointLoader": {"ckpt_name": "checkpoints"},
    "CLIPVisionEncode": {"clip_vision": "CLIP_VISION", "image": "IMAGE", "crop": ["center", "none"]},
    "Hunyuan3Dv2Conditioning": {"clip_vision_output": "CLIP_VISION_OUTPUT"},
    "Hunyuan3Dv2ConditioningMultiView": {"front": "CLIP_VISION_OUTPUT"},
    "EmptyLatentHunyuan3Dv2": {"resolution": int, "batch_size": int},
    "ModelSamplingAuraFlow": {"model": "MODEL", "shift": float},
    "VAEDecodeHunyuan3D": {"samples": "LATENT", "vae": "VAE", "num_chunks": int, "octree_resolution": int},
    "VoxelToMesh": {"voxel": "VOXEL", "algorithm": ["surface net", "basic"], "threshold": float},
    "SaveGLB": {"mesh": "MESH", "filename_prefix": str},
}
LISTS = ("checkpoints", "loras", "vae", "uploads")


class FakeComfy:
    def __init__(self):
        self.models = {"checkpoints": [DEFAULT_CHECKPOINT], "loras": [DEFAULT_LORA, DEFAULT_LCM_LORA],
                       "vae": ["sdxl_vae.safetensors"]}
        self.models_endpoint = True      # False = old ComfyUI without GET /models/<folder>
        self.history_delay = 2           # empty /history answers before a prompt is "done"
        self.fail_with = None            # execution_error data for the next prompts
        self.prompts = []                # workflows received
        self.client_ids = []
        self.uploads = {}                # name -> PIL image
        self.files = {}                  # (type, filename) -> png bytes
        self.history = {}
        self.polls = {}
        self.upload_fields = []

    # -- validation, like ComfyUI's validate_prompt
    def validate(self, wf):
        errors = {}

        def bad(nid, cls, etype, message, details="", **extra):
            errors.setdefault(nid, {"errors": [], "dependent_outputs": [], "class_type": cls})
            errors[nid]["errors"].append({"type": etype, "message": message, "details": details,
                                          "extra_info": extra})

        if not isinstance(wf, dict) or not wf:
            return {"type": "invalid_prompt", "message": "Cannot execute because the prompt is empty",
                    "details": "", "extra_info": {}}, {}
        for nid, node in wf.items():
            cls = node.get("class_type")
            if cls not in INPUTS:
                return {"type": "missing_node_type", "message": f"Node '{nid}' has unknown class_type {cls}",
                        "details": f"Node ID '#{nid}'", "extra_info": {}}, {}
            inputs = node.get("inputs", {})
            for name, kind in INPUTS[cls].items():
                if name not in inputs:
                    bad(nid, cls, "required_input_missing", "Required input is missing", name,
                        input_name=name)
                    continue
                v = inputs[name]
                if isinstance(kind, str) and kind.isupper():  # a link
                    ok = (isinstance(v, list) and len(v) == 2 and str(v[0]) in wf
                          and isinstance(v[1], int)
                          and v[1] < len(OUTPUTS[wf[str(v[0])]["class_type"]])
                          and OUTPUTS[wf[str(v[0])]["class_type"]][v[1]] == kind)
                    if not ok:
                        bad(nid, cls, "return_type_mismatch", "Return type mismatch between linked nodes",
                            f"{name}, {v!r} is not a {kind} output", input_name=name)
                elif kind in LISTS:
                    allowed = list(self.uploads) if kind == "uploads" else self.models[kind]
                    if v not in allowed:
                        bad(nid, cls, "value_not_in_list", "Value not in list",
                            f"{name}: '{v}' not in {allowed}", input_name=name,
                            input_config=[allowed, {}], received_value=v)
                elif isinstance(kind, list):
                    if v not in kind:
                        bad(nid, cls, "value_not_in_list", "Value not in list",
                            f"{name}: '{v}' not in {kind}", input_name=name,
                            input_config=[kind, {}], received_value=v)
                elif kind is float:
                    if isinstance(v, bool) or not isinstance(v, (int, float)):
                        bad(nid, cls, "invalid_input_type", "Failed to convert an input value to a FLOAT",
                            f"{name}, {v!r}", input_name=name)
                elif kind is int:
                    if isinstance(v, bool) or not isinstance(v, int):
                        bad(nid, cls, "invalid_input_type", "Failed to convert an input value to a INT",
                            f"{name}, {v!r}", input_name=name)
                    elif name in ("width", "height") and (v % 8 or v < 16):
                        bad(nid, cls, "value_not_valid", "Value not a multiple of 8", f"{name}: {v}",
                            input_name=name)
                elif not isinstance(v, kind):
                    bad(nid, cls, "invalid_input_type", f"Not a {kind.__name__}", name, input_name=name)
        if not any(n["class_type"] in ("SaveImage", "PreviewImage", "SaveGLB") for n in wf.values()):
            return {"type": "prompt_no_outputs", "message": "Prompt has no outputs", "details": "",
                    "extra_info": {}}, {}
        if errors:
            return {"type": "prompt_outputs_failed_validation", "message": "Prompt outputs failed validation",
                    "details": "", "extra_info": {}}, errors
        return None, {}

    # -- "execution": draw a plain-background sprite-ish picture
    def render(self, wf, pid):
        if any(n["class_type"] == "SaveGLB" for n in wf.values()):
            return self.render_3d(wf)
        sampler = next(n for n in wf.values() if n["class_type"] == "KSampler")
        src = wf[sampler["inputs"]["latent_image"][0]]
        if src["class_type"] == "EmptyLatentImage":
            w, h = src["inputs"]["width"], src["inputs"]["height"]
        else:
            w, h = self.uploads[wf[src["inputs"]["pixels"][0]]["inputs"]["image"]].size
        seed = sampler["inputs"]["seed"]
        rng = np.random.default_rng(seed)
        color = tuple(int(c) for c in rng.integers(20, 200, 3))
        img = Image.new("RGB", (w, h), (255, 255, 255))
        d = ImageDraw.Draw(img)
        d.ellipse([w * 0.3, h * 0.15, w * 0.7, h * 0.9], fill=color)
        d.rectangle([w * 0.42, h * 0.3, w * 0.48, h * 0.36], fill=(10, 10, 10))
        outputs = {}
        for nid, node in wf.items():
            if node["class_type"] in ("SaveImage", "PreviewImage"):
                kind = "output" if node["class_type"] == "SaveImage" else "temp"
                name = f"{node['inputs'].get('filename_prefix', 'ComfyUI')}_{len(self.files) + 1:05}_.png"
                buf = io.BytesIO()
                img.save(buf, "PNG")
                self.files[(kind, name)] = buf.getvalue()
                outputs[nid] = {"images": [{"filename": name, "subfolder": "", "type": kind}]}
        return outputs

    def render_3d(self, wf):
        """A grey ellipsoid shaped like the fake pictures (glTF: y up, front +z)."""
        import trimesh
        mesh = trimesh.creation.icosphere(subdivisions=4)
        mesh.apply_scale([0.4, 0.75, 0.3])
        outputs = {}
        for nid, node in wf.items():
            if node["class_type"] == "SaveGLB":
                sub, _, prefix = node["inputs"]["filename_prefix"].rpartition("/")
                name = f"{prefix}_{len(self.files) + 1:05}_.glb"
                self.files[("output", sub, name)] = mesh.export(file_type="glb")
                outputs[nid] = {"3d": [{"filename": name, "subfolder": sub, "type": "output"}],
                                "camera_info": [None], "model_3d_info": []}
        return outputs

    def finish(self, pid):
        wf = self.prompts[int(pid.split("-")[1])]
        msgs = [["execution_start", {"prompt_id": pid, "timestamp": 1}]]
        if self.fail_with:
            msgs.append(["execution_error", {"prompt_id": pid, **self.fail_with,
                                             "executed": [], "traceback": [], "current_inputs": {},
                                             "current_outputs": []}])
            return {"prompt": [0, pid, wf, {}, []], "outputs": {},
                    "status": {"status_str": "error", "completed": False, "messages": msgs}}
        msgs.append(["execution_success", {"prompt_id": pid, "timestamp": 2}])
        return {"prompt": [0, pid, wf, {}, ["12"]], "outputs": self.render(wf, pid),
                "status": {"status_str": "success", "completed": True, "messages": msgs}}


class Handler(BaseHTTPRequestHandler):
    fake: FakeComfy

    def log_message(self, *args):
        pass

    def send(self, code, payload, ctype="application/json"):
        body = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        fake, url = self.server.fake, urllib.parse.urlsplit(self.path)
        q = dict(urllib.parse.parse_qsl(url.query, keep_blank_values=True))
        path = url.path
        if path == "/system_stats":
            gpu = "cuda:0 NVIDIA GeForce RTX 4080 Laptop GPU : cudaMallocAsync"
            return self.send(200, {"system": {"os": "nt", "comfyui_version": "0.39.0"},
                                   "devices": [{"name": gpu, "type": "cuda", "index": 0,
                                                "vram_total": 12 * 2**30, "vram_free": 11 * 2**30}]})
        if path.startswith("/models/"):
            kind = path.split("/", 2)[2]
            if not fake.models_endpoint or kind not in fake.models:
                return self.send(404, b"404: Not Found", "text/plain")
            return self.send(200, fake.models[kind])
        if path.startswith("/object_info/"):
            node = path.split("/", 2)[2]
            inp = {"CheckpointLoaderSimple": ("ckpt_name", "checkpoints"),
                   "LoraLoader": ("lora_name", "loras"), "VAELoader": ("vae_name", "vae")}.get(node)
            if not inp:
                return self.send(200, {})
            return self.send(200, {node: {"input": {"required": {inp[0]: [fake.models[inp[1]], {}]}},
                                          "output": [], "name": node}})
        if path.startswith("/history/"):
            pid = urllib.parse.unquote(path.split("/", 2)[2])
            if pid not in fake.history:
                fake.polls[pid] = fake.polls.get(pid, 0) + 1
                known = pid in {f"p-{i}" for i in range(len(fake.prompts))}
                if not known or fake.polls[pid] <= fake.history_delay:
                    return self.send(200, {})
                fake.history[pid] = fake.finish(pid)
            return self.send(200, {pid: fake.history[pid]})
        if path == "/view":
            key = (q.get("type", "output"), q.get("filename"))
            if q.get("subfolder"):
                key = (key[0], q["subfolder"], key[1])
            if key not in fake.files:
                return self.send(404, b"404: Not Found", "text/plain")
            return self.send(200, fake.files[key], "image/png")
        return self.send(404, b"404: Not Found", "text/plain")

    def do_POST(self):
        fake = self.server.fake
        body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        if self.path == "/prompt":
            data = json.loads(body)
            error, node_errors = fake.validate(data.get("prompt"))
            if error:
                return self.send(400, {"error": error, "node_errors": node_errors})
            pid = f"p-{len(fake.prompts)}"
            fake.prompts.append(data["prompt"])
            fake.client_ids.append(data.get("client_id"))
            return self.send(200, {"prompt_id": pid, "number": len(fake.prompts), "node_errors": {}})
        if self.path == "/upload/image":
            msg = email.parser.BytesParser(policy=email.policy.HTTP).parsebytes(
                b"Content-Type: " + self.headers["Content-Type"].encode() + b"\r\n\r\n" + body)
            fields, image = {}, None
            for part in msg.iter_parts():
                name = part.get_param("name", header="content-disposition")
                if name == "image":
                    image = (part.get_filename(), part.get_payload(decode=True))
                else:
                    fields[name] = part.get_payload(decode=True).decode()
            fake.upload_fields.append(fields)
            if not image:
                return self.send(400, b"no image", "text/plain")
            img = Image.open(io.BytesIO(image[1]))
            img.load()
            fake.uploads[image[0]] = img
            return self.send(200, {"name": image[0], "subfolder": "", "type": fields.get("type", "input")})
        return self.send(404, b"404: Not Found", "text/plain")


@pytest.fixture
def comfy():
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.fake = FakeComfy()
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True)
    thread.start()
    client = ComfyClient(f"http://127.0.0.1:{server.server_address[1]}", timeout=5, poll_interval=0.01)
    yield server.fake, client
    server.shutdown()
    server.server_close()


@pytest.fixture(autouse=True)
def white_key(monkeypatch):
    """Stand-in for the ONNX cut-out: everything near-white is background."""
    def fake_remove_background(rgb, model="isnet-general-use"):
        return np.where((rgb > 235).all(-1), 0, 255).astype(np.uint8)
    monkeypatch.setattr(pipeline, "remove_background", fake_remove_background)


def ksampler(wf):
    return next(n for n in wf.values() if n["class_type"] == "KSampler")["inputs"]


def by_class(wf, cls):
    return [(nid, n) for nid, n in wf.items() if n["class_type"] == cls]


# --------------------------------------------------------------------------- workflows

def test_txt2img_workflow_wiring(comfy):
    fake, _ = comfy
    wf = build_workflow("pixel art sprite of a fox", "photo", seed=7)
    assert fake.validate(wf) == (None, {})
    (lid, lora), = by_class(wf, "LoraLoader")
    assert lora["inputs"]["lora_name"] == DEFAULT_LORA
    assert lora["inputs"]["model"] == ["1", 0] and lora["inputs"]["clip"] == ["1", 1]
    k = ksampler(wf)
    assert k["model"] == [lid, 0]                       # sampler uses the LoRA-patched model
    for _, enc in by_class(wf, "CLIPTextEncode"):
        assert enc["inputs"]["clip"] == [lid, 1]        # ... and the LoRA-patched CLIP
    latent = wf[k["latent_image"][0]]
    assert latent["class_type"] == "EmptyLatentImage" and latent["inputs"]["width"] == 1024
    assert k["denoise"] == 1.0 and k["seed"] == 7
    assert by_class(wf, "SaveImage")
    json.dumps(wf)                                      # plain JSON


def test_img2img_fast_and_vae_wiring(comfy):
    fake, _ = comfy
    fake.uploads["sketch.png"] = Image.new("RGB", (64, 64))
    wf = build_workflow("x", "", seed=1, image="sketch.png", denoise=0.45,
                        lcm_lora=DEFAULT_LCM_LORA, vae="sdxl_vae.safetensors",
                        sampler="lcm", scheduler="sgm_uniform", steps=8, cfg=1.5)
    assert fake.validate(wf) == (None, {})
    assert not by_class(wf, "EmptyLatentImage")
    k = ksampler(wf)
    enc = wf[k["latent_image"][0]]
    assert enc["class_type"] == "VAEEncode" and k["denoise"] == 0.45
    assert wf[enc["inputs"]["pixels"][0]]["inputs"]["image"] == "sketch.png"
    (vid, _), = by_class(wf, "VAELoader")
    assert enc["inputs"]["vae"] == [vid, 0]
    assert by_class(wf, "VAEDecode")[0][1]["inputs"]["vae"] == [vid, 0]
    # pixel LoRA -> LCM LoRA -> ModelSamplingDiscrete(lcm) -> KSampler
    msd = wf[k["model"][0]]
    assert msd["class_type"] == "ModelSamplingDiscrete" and msd["inputs"]["sampling"] == "lcm"
    lcm = wf[msd["inputs"]["model"][0]]
    assert lcm["inputs"]["lora_name"] == DEFAULT_LCM_LORA
    assert wf[lcm["inputs"]["model"][0]]["inputs"]["lora_name"] == DEFAULT_LORA


def test_no_lora_workflow_is_valid(comfy):
    fake, _ = comfy
    wf = build_workflow("x", "", seed=1, lora=None)
    assert fake.validate(wf) == (None, {})
    assert not by_class(wf, "LoraLoader") and ksampler(wf)["model"] == ["1", 0]


def test_generation_size_and_prompts():
    assert generation_size() == (1024, 1024)
    assert generation_size(32 / 48) == (832, 1216)       # person preset
    assert generation_size(160 / 90) == (1344, 768)      # background preset
    assert generation_size(None, 1000, 700) == (1000, 704)
    pos, neg = sprite_prompt("a knight with a red cape.", "person", system="snes", size=(16, 24))
    assert pos.startswith("pixel art sprite of a knight with a red cape,")
    assert "16-bit SNES" in pos and "chunky pixels" in pos and "full body" in pos
    assert "3d render" in neg and "multiple views" in neg
    pos, _ = sprite_prompt("pixel art of a forest lake", "background", size=(160, 90))
    assert pos.startswith("pixel art landscape background of a forest lake")
    with pytest.raises(ValueError):
        sprite_prompt("  ", "person")
    with pytest.raises(ValueError):
        sprite_prompt("x", "vehicle")


def test_prepare_init_image_nearest_upscale():
    sketch = np.zeros((48, 32, 4), np.uint8)
    sketch[8:40, 8:24] = (200, 30, 30, 255)
    sketch[20, 12] = (0, 0, 255, 255)
    img = prepare_init_image(sketch, 832, 1216)
    assert img.size == (832, 1216) and img.mode == "RGB"
    arr = np.asarray(img)
    # whole-number scale 25 -> 800x1200, centred (16 px / 8 px margins), transparent -> white
    assert (arr[:8] == 255).all() and (arr[:, :16] == 255).all()
    assert (arr[8 + 20 * 25: 8 + 21 * 25, 16 + 12 * 25: 16 + 13 * 25] == (0, 0, 255)).all()
    assert (arr[8 + 9 * 25, 16 + 9 * 25] == (200, 30, 30)).all()


# --------------------------------------------------------------------------- client

def test_ping_and_models(comfy):
    fake, client = comfy
    assert client.ping()["devices"][0]["vram_total"] == 12 * 2**30
    assert client.list_models("loras") == [DEFAULT_LORA, DEFAULT_LCM_LORA]
    fake.models_endpoint = False                        # older ComfyUI: /object_info fallback
    assert client.list_models("checkpoints") == [DEFAULT_CHECKPOINT]
    assert client.list_models("vae") == ["sdxl_vae.safetensors"]


def test_comfyui_not_running():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    client = ComfyClient(f"http://127.0.0.1:{port}", timeout=15)  # Windows: refused takes ~2 s
    with pytest.raises(ComfyUIUnavailable, match="run_nvidia_gpu.bat"):
        client.ping()
    with pytest.raises(ComfyUIUnavailable, match=f"isn't running at http://127.0.0.1:{port}"):
        generate_sprite("a fox", settings_for("animal"), client=client)


def test_comfyui_folder_and_command(tmp_path, monkeypatch):
    from pixel_pipeline.generate import comfyui_command, find_comfyui
    (tmp_path / "run_nvidia_gpu.bat").write_text(
        ".\\python_embeded\\python.exe -s ComfyUI\\main.py --windows-standalone-build --lowvram\r\npause\r\n")
    monkeypatch.setenv("COMFYUI_DIR", str(tmp_path))
    assert find_comfyui() == tmp_path
    cmd = comfyui_command(tmp_path)
    assert cmd[0].endswith("python.exe") and cmd[1:] == [
        "-s", "ComfyUI\\main.py", "--windows-standalone-build", "--lowvram", "--disable-auto-launch"]


def test_ensure_running_starts_comfyui_at_the_default_address(comfy, tmp_path, monkeypatch):
    import pixel_pipeline.generate as g
    _, running = comfy
    started = []
    monkeypatch.setattr(g, "find_comfyui", lambda: tmp_path)
    monkeypatch.setattr(g, "start_comfyui", started.append)
    monkeypatch.setattr(g.time, "sleep", lambda s: None)
    client = ComfyClient(g.DEFAULT_URL, timeout=2)
    real_ping, calls = ComfyClient.ping, []

    def ping(self):                      # down until started, then the fake server answers
        calls.append(1)
        if not started:
            raise ComfyUIUnavailable("down")
        return real_ping(running)
    monkeypatch.setattr(ComfyClient, "ping", ping)
    assert "system" in client.ensure_running()
    assert started == [tmp_path]
    # any other address is never started
    other = ComfyClient("http://127.0.0.1:8000")
    started.clear()
    with pytest.raises(ComfyUIUnavailable):
        other.ensure_running()
    assert not started


def test_started_comfyui_is_hidden_logged_and_stopped(tmp_path, monkeypatch):
    import sys
    import pixel_pipeline.generate as g
    monkeypatch.setattr(g, "COMFYUI_LOG", tmp_path / "comfyui.log")
    monkeypatch.setattr(g, "comfyui_command", lambda folder: [
        sys.executable, "-c", "import time; print('hello from comfy', flush=True); time.sleep(60)"])
    proc = g.start_comfyui(tmp_path)
    deadline = time.monotonic() + 20
    while "hello from comfy" not in g._log_tail() and time.monotonic() < deadline:
        time.sleep(0.1)
    assert "hello from comfy" in g._log_tail() and proc.poll() is None
    g.stop_started_comfyui()
    assert proc.poll() is not None and not g._started


def test_env_var_sets_url(monkeypatch):
    monkeypatch.setenv("COMFYUI_URL", "127.0.0.1:8000")
    assert ComfyClient().base_url == "http://127.0.0.1:8000"


def test_rejected_workflow_lists_installed_models(comfy):
    fake, client = comfy
    fake.models["checkpoints"] = ["dreamshaperXL.safetensors", "sdxl\\juggernautXL.safetensors"]
    with pytest.raises(ComfyUIModelMissing) as e:
        client.queue(build_workflow("x", "", seed=1, checkpoint="missing.safetensors"))
    msg = str(e.value)
    assert "CheckpointLoaderSimple (node 1): model file 'missing.safetensors' is not installed" in msg
    assert "models\\checkpoints" in msg
    assert "dreamshaperXL.safetensors, sdxl\\juggernautXL.safetensors" in msg
    assert e.value.kind == "checkpoints" and e.value.name == "missing.safetensors"
    # other validation errors are readable too
    with pytest.raises(ComfyUIWorkflowError, match="KSampler .node 10.: Value not in list"):
        client.queue(build_workflow("x", "", seed=1, sampler="nope"))
    with pytest.raises(ComfyUIWorkflowError, match="return_type_mismatch|Return type mismatch"):
        wf = build_workflow("x", "", seed=1)
        wf["11"]["inputs"]["samples"] = ["1", 0]
        client.queue(wf)


def test_missing_lora_detected_before_queueing(comfy):
    fake, client = comfy
    fake.models["loras"] = []
    with pytest.raises(ComfyUIModelMissing) as e:
        generate_sprite("a fox", settings_for("animal"), client=client)
    assert "pixel-art-xl.safetensors" in str(e.value) and "models\\loras" in str(e.value)
    assert "huggingface.co/nerijs/pixel-art-xl" in str(e.value) and "(none)" in str(e.value)
    assert not fake.prompts
    ok, report = check_setup(client)
    assert not ok and "MISSING" in report and "RTX 4080 Laptop GPU, 12.0 GB VRAM" in report


def test_auto_picks_similar_default_models(comfy):
    fake, client = comfy
    fake.models["checkpoints"] = ["other.safetensors", "sdxl\\sd_xl_base_1.0_0.9vae.safetensors"]
    fake.models["loras"] = ["styles\\Pixel_Art_XL_v1.1.safetensors"]
    generate_sprite("a fox", settings_for("animal"), client=client, seed=3)
    wf = fake.prompts[0]
    assert wf["1"]["inputs"]["ckpt_name"] == "sdxl\\sd_xl_base_1.0_0.9vae.safetensors"
    assert by_class(wf, "LoraLoader")[0][1]["inputs"]["lora_name"] == "styles\\Pixel_Art_XL_v1.1.safetensors"
    ok, report = check_setup(client)
    assert ok and "Ready." in report
    # an explicitly requested model is never swapped
    with pytest.raises(ComfyUIModelMissing):
        generate_sprite("a fox", settings_for("animal"), client=client, checkpoint="my_model.safetensors")


def test_fp16_fix_vae_used_when_installed(comfy):
    fake, client = comfy
    generate_sprite("a fox", settings_for("animal"), client=client, seed=3)
    (_, vl), = by_class(fake.prompts[0], "VAELoader")
    assert vl["inputs"]["vae_name"] == "sdxl_vae.safetensors"
    fake.models["vae"] = []                             # not installed: the checkpoint's own VAE
    generate_sprite("a fox", settings_for("animal"), client=client, seed=3)
    assert not by_class(fake.prompts[1], "VAELoader")


def test_execution_error_out_of_memory(comfy):
    fake, client = comfy
    fake.fail_with = {"node_id": "10", "node_type": "KSampler", "exception_type": "torch.OutOfMemoryError",
                      "exception_message": "CUDA out of memory. Tried to allocate 2.00 GiB"}
    with pytest.raises(ComfyUIWorkflowError) as e:
        generate_sprite("a fox", settings_for("animal"), client=client)
    assert "KSampler (node 10): torch.OutOfMemoryError: CUDA out of memory" in str(e.value)
    assert "--lowvram" in str(e.value)


def test_wait_times_out(comfy):
    fake, client = comfy
    fake.history_delay = 10**9
    pid = client.queue(build_workflow("x", "", seed=1))
    with pytest.raises(ComfyUIError, match="didn't finish within"):
        client.wait(pid, timeout=0.05)


# --------------------------------------------------------------------------- generation

def test_generate_sprite_txt2img(comfy):
    fake, client = comfy
    s = settings_for("person", width=24, height=32, palette="pico-8")
    results = generate_sprite("a knight with a red cape", s, client=client, subject="person",
                              system="snes", seed=1234)
    assert len(results) == 1 and isinstance(results[0], Generated)
    sprite, raw, seed = results[0]
    assert seed == 1234 and sprite.size == (24, 32)
    assert raw.size == (896, 1152)                     # SDXL size nearest the sprite's 3:4
    assert (sprite.rgba[..., 3] == 0).any() and (sprite.rgba[..., 3] == 255).any()  # cut out
    pico = {(0, 0, 0), (29, 43, 83), (126, 37, 83), (0, 135, 81), (171, 82, 54), (95, 87, 79),
            (194, 195, 199), (255, 241, 232), (255, 0, 77), (255, 163, 0), (255, 236, 39),
            (0, 228, 54), (41, 173, 255), (131, 118, 156), (255, 119, 168), (255, 204, 170)}
    assert {tuple(c) for c in sprite.rgba[sprite.rgba[..., 3] == 255][:, :3]} <= pico
    wf = fake.prompts[0]
    assert fake.validate(wf) == (None, {})
    texts = [n["inputs"]["text"] for _, n in by_class(wf, "CLIPTextEncode")]
    assert texts[0].startswith("pixel art sprite of a knight with a red cape") and "SNES" in texts[0]
    assert "3d render" in texts[1]
    k = ksampler(wf)
    assert (k["seed"], k["steps"], k["cfg"], k["sampler_name"], k["scheduler"]) == \
        (1234, 25, 6.0, "euler_ancestral", "normal")
    assert fake.client_ids[0] == client.client_id


def test_count_and_seeds(comfy):
    fake, client = comfy
    results = generate_sprite("a red apple", settings_for("object", width=16, height=16),
                              client=client, seed=41, count=3)
    assert [r.seed for r in results] == [41, 42, 43]
    assert [ksampler(wf)["seed"] for wf in fake.prompts] == [41, 42, 43]
    assert all(r.sprite.size == (16, 16) for r in results)
    # different seeds -> different pictures (the fake colours by seed)
    assert len({r.image.getpixel((512, 512)) for r in results}) == 3
    # random seed when none given
    (_, _, s1), = generate_sprite("a red apple", settings_for("object"), client=client)
    assert ksampler(fake.prompts[-1])["seed"] == s1


def test_fast_mode_and_options(comfy):
    fake, client = comfy
    generate_sprite("a fox", settings_for("animal"), client=client, seed=1, fast=True,
                    width=768, height=768, negative="", vae="sdxl_vae.safetensors", lora_strength=0.8)
    wf = fake.prompts[0]
    k = ksampler(wf)
    assert (k["steps"], k["cfg"], k["sampler_name"], k["scheduler"]) == (8, 1.5, "lcm", "sgm_uniform")
    assert by_class(wf, "ModelSamplingDiscrete") and by_class(wf, "VAELoader")
    assert by_class(wf, "EmptyLatentImage")[0][1]["inputs"]["width"] == 768
    assert by_class(wf, "CLIPTextEncode")[1][1]["inputs"]["text"] == ""
    assert by_class(wf, "LoraLoader")[0][1]["inputs"]["strength_model"] == 0.8
    with pytest.raises(TypeError, match="Unknown generation option"):
        generate_sprite("a fox", settings_for("animal"), client=client, stepz=3)


def test_generate_sprite_img2img(comfy):
    fake, client = comfy
    sketch = np.zeros((48, 32, 4), np.uint8)
    sketch[6:46, 8:24] = (40, 90, 200, 255)
    s = settings_for("person", width=32, height=48)
    (sprite, raw, _), = generate_sprite("a blue wizard", s, client=client, subject="person",
                                           init_image=sketch, denoise=0.55, seed=5)
    assert sprite.size == (32, 48) and raw.size == (832, 1216)
    (name, up), = fake.uploads.items()
    assert up.size == (832, 1216) and name.startswith("pixel-pipeline-init-")
    assert fake.upload_fields[0] == {"overwrite": "true", "type": "input"}
    wf = fake.prompts[0]
    assert fake.validate(wf) == (None, {})
    k = ksampler(wf)
    assert k["denoise"] == 0.55 and wf[k["latent_image"][0]]["class_type"] == "VAEEncode"
    assert by_class(wf, "LoadImage")[0][1]["inputs"]["image"] == name
    with pytest.raises(ValueError, match="denoise"):
        generate_sprite("x", s, client=client, init_image=sketch, denoise=0)


def test_background_subject(comfy):
    fake, client = comfy
    s = settings_for("background", width=64, height=36)
    (sprite, raw, _), = generate_sprite("a misty pine forest", s, client=client, seed=9)
    assert sprite.size == (64, 36) and raw.size == (1344, 768)
    text = by_class(fake.prompts[0], "CLIPTextEncode")[0][1]["inputs"]["text"]
    assert text.startswith("pixel art landscape background of a misty pine forest")  # guessed subject


def test_generate_images_raw_prompt(comfy):
    fake, client = comfy
    (img, seed), = generate_images("exact prompt", "exact negative", client=client, seed=2)
    assert isinstance(img, Image.Image) and seed == 2
    texts = [n["inputs"]["text"] for _, n in by_class(fake.prompts[0], "CLIPTextEncode")]
    assert texts == ["exact prompt", "exact negative"]
    generate_sprite("exact prompt", settings_for("object"), client=client, shape_prompt=False)
    assert by_class(fake.prompts[1], "CLIPTextEncode")[0][1]["inputs"]["text"] == "exact prompt"


# --------------------------------------------------------------------------- templates

def test_fill_template_types_and_errors():
    wf = {"3": {"class_type": "KSampler", "inputs": {"seed": "{{seed}}", "cfg": "{{ cfg }}"}},
          "6": {"class_type": "CLIPTextEncode", "inputs": {"text": "masterpiece, {{prompt}}"}}}
    assert template_placeholders(wf) == {"seed", "cfg", "prompt"}
    out = fill_template(wf, {"seed": 5, "cfg": 2.5, "prompt": 'a "quoted" fox'})
    assert out["3"]["inputs"] == {"seed": 5, "cfg": 2.5}
    assert out["6"]["inputs"]["text"] == 'masterpiece, a "quoted" fox'
    assert wf["3"]["inputs"]["seed"] == "{{seed}}"     # template untouched
    with pytest.raises(ValueError, match="Unknown template placeholder"):
        fill_template({"1": {"class_type": "X", "inputs": {"a": "{{colour}}"}}}, {})
    with pytest.raises(ValueError, match="init image"):
        fill_template({"1": {"class_type": "LoadImage", "inputs": {"image": "{{image}}"}}}, {"image": None})


def test_load_template_rejects_ui_format(tmp_path):
    ui = tmp_path / "ui.json"
    ui.write_text(json.dumps({"nodes": [], "links": [], "version": 0.4}))
    with pytest.raises(ValueError, match="Export \\(API\\)"):
        load_template(ui)
    with pytest.raises(FileNotFoundError):
        load_template(tmp_path / "nope.json")
    body = tmp_path / "body.json"
    body.write_text(json.dumps({"prompt": {"1": {"class_type": "SaveImage", "inputs": {}}}}))
    assert list(load_template(body)) == ["1"]


def test_shipped_templates_txt2img(comfy):
    fake, client = comfy
    path = TEMPLATES / "sdxl_pixel_art_txt2img.json"
    assert {"prompt", "negative", "seed", "width", "height"} <= template_placeholders(load_template(path))
    results = generate_sprite("a green slime", settings_for("animal"), client=client, seed=11,
                              count=2, template=str(path))
    assert [r.seed for r in results] == [11, 12] and all(r.sprite.size == (32, 32) for r in results)
    wf = fake.prompts[0]
    assert fake.validate(wf) == (None, {})
    assert ksampler(wf)["seed"] == 11 and isinstance(ksampler(wf)["steps"], int)
    text = by_class(wf, "CLIPTextEncode")[0][1]["inputs"]["text"]
    assert text.startswith("pixel art sprite of a green slime")
    with pytest.raises(ValueError, match="no \\{\\{image\\}\\} placeholder"):
        generate_sprite("x", settings_for("animal"), client=client, template=str(path),
                        init_image=np.zeros((8, 8, 4), np.uint8))


def test_shipped_templates_img2img(comfy):
    fake, client = comfy
    path = TEMPLATES / "sdxl_pixel_art_img2img.json"
    sketch = np.full((32, 32, 4), (120, 60, 20, 255), np.uint8)
    (_, raw, _), = generate_sprite("a wooden crate", settings_for("object"), client=client,
                                        template=str(path), init_image=sketch, denoise=0.7)
    wf = fake.prompts[0]
    assert fake.validate(wf) == (None, {})
    assert ksampler(wf)["denoise"] == 0.7 and raw.size == (1024, 1024)
    with pytest.raises(ValueError, match="img2img"):
        generate_sprite("x", settings_for("object"), client=client, template=str(path))


def test_template_preview_image_output(comfy, tmp_path):
    """A custom template that only previews (temp images) still works."""
    _, client = comfy
    wf = build_workflow("{{prompt}}", "{{negative}}", seed="{{seed}}", width=512, height=512)
    wf["12"] = {"class_type": "PreviewImage", "inputs": {"images": ["11", 0]}}
    path = tmp_path / "preview_api.json"
    path.write_text(json.dumps(wf))
    (_, raw, _), = generate_sprite("a key", settings_for("object"), client=client, template=str(path))
    assert raw.size == (512, 512)
