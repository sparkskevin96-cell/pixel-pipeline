"""CLI `generate` against the fake ComfyUI server from test_generate."""
from PIL import Image

from pixel_pipeline.cli import main
from test_generate import comfy, white_key  # noqa: F401  (pytest fixtures)


def test_cli_generate_saves_sprites_raw_and_seeds(comfy, tmp_path, capsys):
    fake, client = comfy
    out = tmp_path / "knight.png"
    code = main(["generate", "a knight with a red cape", "-s", "person", "--system", "snes",
                 "--count", "2", "--seed", "7", "--comfyui", client.base_url, "-o", str(out),
                 "--aseprite"])
    assert code == 0, capsys.readouterr().err
    text = capsys.readouterr().out
    assert "seed 7:" in text and "seed 8:" in text
    for i in (1, 2):
        sprite = tmp_path / f"knight-{i}.png"
        assert Image.open(sprite).size == (48, 64)          # readable person size
        assert (tmp_path / f"knight-{i}.raw.png").exists()
        assert (tmp_path / f"knight-{i}.aseprite").exists()
    assert len(fake.prompts) == 2


def test_cli_generate_check_and_errors(comfy, capsys):
    fake, client = comfy
    assert main(["generate", "--check", "--comfyui", client.base_url]) == 0
    assert "ComfyUI" in capsys.readouterr().out
    assert main(["generate", "a fox", "--comfyui", "http://127.0.0.1:9"]) == 2
    assert "run_nvidia_gpu" in capsys.readouterr().err
