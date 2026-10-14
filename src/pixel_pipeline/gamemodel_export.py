"""Engine exports of game models (gamemodel.py's model_game/ folder).

    godot-3d-model   <name>.glb (LOD0) + <name>.glb.import (looping clips loop; Godot keeps
                     our _subresources and adds its own LODs) + <name>_lod1/_lod2.glb; static
                     models also get <name>.tscn: the three LODs with visibility ranges.
    unity-3d-model   <name>.ppmodel + textures + Editor/PixelPipelineModelImporter.cs: a
                     ScriptedImporter (no packages) that builds the prefab: LODGroup, skinned
                     or plain renderers, materials (URP Lit or Standard), the skeleton and every
                     clip (loop flags set), plus <name>.controller with a state per clip.
                     Static models also get lod0.obj/.mtl (Unity opens those natively).
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

from .gamemodel import FOLDER, _stem

LOOPING = {"idle", "walk", "run", "jog", "sprint", "happy", "swim", "fly"}


def model_folder(src) -> Path:
    """model_game/ of an exports.Source / asset / folder; builds it if missing."""
    folder = Path(getattr(src, "folder", src))
    mg = folder / FOLDER
    if not (mg / "lod0.glb").exists():
        from .gamemodel import game_model_for
        game_model_for(_AssetLike(folder))
    return mg


class _AssetLike:
    def __init__(self, folder: Path):
        self.folder = folder
        self.name = folder.name
        meta = folder / "asset.json"
        self.kind, self.extra = "prop", {}
        for m in ("asset.json", "character.json"):
            if (folder / m).exists():
                try:
                    d = json.loads((folder / m).read_text(encoding="utf-8"))
                except ValueError:
                    continue
                self.name = d.get("name", self.name)
                self.kind = "character" if m == "character.json" else d.get("kind", "prop")
                self.extra = d.get("extra", {})
                if m == "character.json" and d.get("kind") == "animal":
                    self.extra["body"] = "animal"
                break
        del meta


def _info(mg: Path) -> dict:
    return json.loads((mg / "info.json").read_text(encoding="utf-8"))


def _loops(mg: Path) -> dict:
    """{clip: loops?} from the source rigged.glb's extras (else by name)."""
    info = _info(mg)
    out = {}
    src = mg.parent / info.get("source", "rigged.glb")
    doc = None
    if src.exists() and src.suffix == ".glb":
        from .anim3d import Gltf
        doc = Gltf(src).doc
    for a in (doc or {}).get("animations", []):
        name = a.get("name", "")
        loop = (a.get("extras") or {}).get("loop")
        out[name] = bool(loop) if loop is not None else name.split("_")[0] in LOOPING
    for c in info.get("clips", []):
        out.setdefault(c, c.split("_")[0] in LOOPING)
    return out


def godot_import_file(glb_name: str, loops: dict) -> str:
    anims = ",\n".join(f'"{n}": {{\n"settings/loop_mode": {1 if l else 0}\n}}' for n, l in loops.items())
    return ("[remap]\n\nimporter=\"scene\"\nimporter_version=1\ntype=\"PackedScene\"\n\n"
            f"[deps]\n\nsource_file=\"res://{glb_name}\"\n\n[params]\n\n"
            "_subresources={\n\"animations\": {\n" + anims + "\n}\n}\n")


def godot_lod_scene(stem: str, ranges=(0.0, 25.0, 60.0)) -> str:
    """Three LOD instances switching by camera distance (visibility ranges, metres)."""
    lines = [f'[gd_scene load_steps=4 format=3]', ""]
    for i in range(3):
        f = f"{stem}.glb" if i == 0 else f"{stem}_lod{i}.glb"
        lines.append(f'[ext_resource type="PackedScene" path="{f}" id="{i + 1}"]')
    lines += ["", f'[node name="{stem}" type="Node3D"]', ""]
    for i in range(3):
        begin, end = ranges[i], (ranges[i + 1] if i < 2 else 0.0)
        lines.append(f'[node name="LOD{i}" parent="." instance=ExtResource("{i + 1}")]')
        lines.append(f"visibility_range_begin = {begin}")
        if end:
            lines.append(f"visibility_range_end = {end}")
        lines.append("")
    return "\n".join(lines)


def export_godot(src, out: Path, **_) -> list[Path]:
    mg = model_folder(src)
    info = _info(mg)
    stem = _stem(info["name"])
    out.mkdir(parents=True, exist_ok=True)
    written = []
    for i, lod in enumerate(info["lods"]):
        dst = out / (f"{stem}.glb" if i == 0 else f"{stem}_lod{i}.glb")
        shutil.copyfile(mg / lod["file"], dst)
        written.append(dst)
        if info.get("skinned"):
            imp = dst.with_name(dst.name + ".import")
            imp.write_text(godot_import_file(dst.name, _loops(mg)), encoding="utf-8")
            written.append(imp)
    if not info.get("skinned") and len(info["lods"]) >= 3:
        tscn = out / f"{stem}.tscn"
        tscn.write_text(godot_lod_scene(stem), encoding="utf-8")
        written.append(tscn)
    for extra in ("preview.png", "preview.gif"):
        if (mg / extra).exists():
            shutil.copyfile(mg / extra, out / f"{stem}_{extra}")
            written.append(out / f"{stem}_{extra}")
    readme = out / f"{stem}_README.txt"
    readme.write_text(
        f"{info['name']} - game model from pixel-pipeline ({info['look']} look)\n\n"
        f"{stem}.glb is LOD0 ({info['lods'][0]['tris']} triangles); _lod1/_lod2 are lighter.\n"
        + ("Skinned with " + str(info['bones']) + " bones; clips: " + ", ".join(info['clips'])
           + ".\nThe .import file makes looping clips loop. Drag the .glb into a scene; its\n"
           "AnimationPlayer has every clip.\n" if info.get("skinned") else
           f"{stem}.tscn switches LODs by distance (visibility ranges).\n"),
        encoding="utf-8")
    written.append(readme)
    return written


def export_unity(src, out: Path, **_) -> list[Path]:
    mg = model_folder(src)
    info = _info(mg)
    stem = _stem(info["name"])
    out.mkdir(parents=True, exist_ok=True)
    written = []
    pp = next(mg.glob("*.ppmodel"), None)
    if pp is None:
        raise ValueError(f"{mg} has no .ppmodel (rebuild the game model)")
    data = json.loads(pp.read_text(encoding="utf-8"))
    for lod in data["lods"]:
        for key in ("albedo", "normal"):
            if lod.get(key):
                new = f"{stem}_{lod[key]}"
                shutil.copyfile(mg / lod[key], out / new)
                written.append(out / new)
                lod[key] = new
    data["name"] = stem
    dst = out / f"{stem}.ppmodel"
    dst.write_text(json.dumps(data, separators=(",", ":")), encoding="utf-8")
    written.append(dst)
    if not info.get("skinned") and (mg / "lod0.obj").exists():
        for f in ("lod0.obj", "lod0.mtl"):
            text = (mg / f).read_text(encoding="utf-8").replace("lod0", f"{stem}")
            p = out / f.replace("lod0", stem)
            p.write_text(text, encoding="utf-8")
            written.append(p)
        written.append(Path(shutil.copyfile(mg / "lod0_albedo.png", out / f"{stem}_albedo.png")))
    ed = out / "Editor"
    ed.mkdir(exist_ok=True)
    cs = ed / "PixelPipelineModelImporter.cs"
    cs.write_text(IMPORTER_CS, encoding="utf-8")
    written.append(cs)
    readme = out / f"{stem}_README.txt"
    readme.write_text(
        f"{info['name']} - game model from pixel-pipeline for Unity ({info['look']} look)\n\n"
        "Copy this folder into Assets/. Editor/PixelPipelineModelImporter.cs imports the\n"
        f"{stem}.ppmodel file as a prefab: LODGroup with {len(info['lods'])} LODs, materials\n"
        "(URP Lit when URP is active, else Standard; pixel look = Point filtering)"
        + (f", {info['bones']} bones and the clips {', '.join(info['clips'])}.\n"
           f"{stem}.controller (made next to it on import) has a state per clip; set it on the\n"
           "prefab's Animator.\n" if info.get("skinned") else
           f".\n{stem}.obj also opens without the importer.\n")
        + "No packages needed (glTFast is not required; the .glb files in model_game/ work with\n"
          "it if you prefer).\n", encoding="utf-8")
    written.append(readme)
    return written


IMPORTER_CS = r'''// pixel-pipeline game model importer: .ppmodel -> prefab (no packages needed).
// glTF space in the file (right-handed); x is mirrored into Unity's left-handed space.
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using UnityEditor;
using UnityEditor.Animations;
using UnityEditor.AssetImporters;
using UnityEngine;
using UnityEngine.Rendering;

namespace PixelPipeline
{
    [Serializable] public class PPCurve { public int bone; public string property; public float[] times; public float[] values; public bool step; }
    [Serializable] public class PPClip { public string name; public bool loop; public float length; public PPCurve[] curves; }
    [Serializable] public class PPBone { public string name; public int parent; public float[] position; public float[] rotation; }
    [Serializable] public class PPLod { public string albedo; public string normal; public float[] vertices; public float[] normals; public float[] uv; public float[] tangents; public int[] triangles; public int[] boneIndices; public float[] boneWeights; }
    [Serializable] public class PPModel { public string format; public int version; public string name; public string kind; public string look; public string filter; public bool skinned; public PPLod[] lods; public PPBone[] bones; public int[] skinBones; public PPClip[] clips; }

    [ScriptedImporter(3, "ppmodel")]
    public class PixelPipelineModelImporter : ScriptedImporter
    {
        public float[] lodScreenHeights = { 0.4f, 0.15f, 0.03f };

        static Texture2D LoadTexture(string path, bool linear, bool point, string name)
        {
            var tex = new Texture2D(2, 2, TextureFormat.RGBA32, true, linear);
            tex.LoadImage(File.ReadAllBytes(path));
            tex.name = name;
            tex.filterMode = point ? FilterMode.Point : FilterMode.Bilinear;
            tex.wrapMode = TextureWrapMode.Clamp;
            tex.anisoLevel = point ? 0 : 4;
            tex.Apply(true, false);
            return tex;
        }

        static Shader LitShader(out bool urp)
        {
            urp = GraphicsSettings.currentRenderPipeline != null;
            Shader s = urp ? Shader.Find("Universal Render Pipeline/Lit") : null;
            if (s == null) { urp = false; s = Shader.Find("Standard"); }
            return s;
        }

        static string PathTo(Transform t, Transform root)
        {
            var parts = new List<string>();
            while (t != null && t != root) { parts.Insert(0, t.name); t = t.parent; }
            return string.Join("/", parts);
        }

        public override void OnImportAsset(AssetImportContext ctx)
        {
            var data = JsonUtility.FromJson<PPModel>(File.ReadAllText(ctx.assetPath));
            string dir = Path.GetDirectoryName(ctx.assetPath);
            bool point = data.filter == "point";
            var root = new GameObject(string.IsNullOrEmpty(data.name) ? Path.GetFileNameWithoutExtension(ctx.assetPath) : data.name);
            Transform[] bones = new Transform[data.bones != null ? data.bones.Length : 0];
            for (int i = 0; i < bones.Length; i++) bones[i] = new GameObject(data.bones[i].name).transform;
            for (int i = 0; i < bones.Length; i++)
            {
                var b = data.bones[i];
                bones[i].SetParent(b.parent >= 0 ? bones[b.parent] : root.transform, false);
                bones[i].localPosition = new Vector3(-b.position[0], b.position[1], b.position[2]);
                bones[i].localRotation = new Quaternion(b.rotation[0], -b.rotation[1], -b.rotation[2], b.rotation[3]);
            }
            bool urp;
            Shader shader = LitShader(out urp);
            var lods = new List<LOD>();
            for (int k = 0; k < data.lods.Length; k++)
            {
                var L = data.lods[k];
                int n = L.vertices.Length / 3;
                var v = new Vector3[n]; var nr = new Vector3[n]; var uv = new Vector2[n];
                for (int i = 0; i < n; i++)
                {
                    v[i] = new Vector3(-L.vertices[3 * i], L.vertices[3 * i + 1], L.vertices[3 * i + 2]);
                    nr[i] = new Vector3(-L.normals[3 * i], L.normals[3 * i + 1], L.normals[3 * i + 2]);
                    uv[i] = new Vector2(L.uv[2 * i], 1f - L.uv[2 * i + 1]);
                }
                var tri = new int[L.triangles.Length];
                for (int i = 0; i < tri.Length; i += 3) { tri[i] = L.triangles[i]; tri[i + 1] = L.triangles[i + 2]; tri[i + 2] = L.triangles[i + 1]; }
                var mesh = new Mesh { name = data.name + "_LOD" + k };
                if (n > 65000) mesh.indexFormat = IndexFormat.UInt32;
                mesh.vertices = v; mesh.normals = nr; mesh.uv = uv; mesh.triangles = tri;
                if (L.tangents != null && L.tangents.Length == 4 * n)
                {
                    var tg = new Vector4[n];
                    for (int i = 0; i < n; i++) tg[i] = new Vector4(-L.tangents[4 * i], L.tangents[4 * i + 1], L.tangents[4 * i + 2], -L.tangents[4 * i + 3]);
                    mesh.tangents = tg;
                }
                else mesh.RecalculateTangents();
                var mat = new Material(shader) { name = data.name + "_LOD" + k };
                var albedo = LoadTexture(Path.Combine(dir, L.albedo), false, point, data.name + "_albedo" + k);
                ctx.AddObjectToAsset("albedo" + k, albedo);
                mat.SetTexture(urp ? "_BaseMap" : "_MainTex", albedo);
                mat.SetFloat(urp ? "_Smoothness" : "_Glossiness", 0.1f);
                mat.SetFloat("_Metallic", 0f);
                if (!string.IsNullOrEmpty(L.normal) && File.Exists(Path.Combine(dir, L.normal)))
                {
                    var nm = LoadTexture(Path.Combine(dir, L.normal), true, point, data.name + "_normal" + k);
                    ctx.AddObjectToAsset("normal" + k, nm);
                    mat.SetTexture("_BumpMap", nm);
                    mat.EnableKeyword("_NORMALMAP");
                }
                ctx.AddObjectToAsset("material" + k, mat);
                var go = new GameObject("LOD" + k);
                go.transform.SetParent(root.transform, false);
                Renderer r;
                if (data.skinned && L.boneIndices != null && L.boneIndices.Length == 4 * n)
                {
                    var bw = new BoneWeight[n];
                    for (int i = 0; i < n; i++)
                    {
                        bw[i] = new BoneWeight
                        {
                            boneIndex0 = L.boneIndices[4 * i], weight0 = L.boneWeights[4 * i],
                            boneIndex1 = L.boneIndices[4 * i + 1], weight1 = L.boneWeights[4 * i + 1],
                            boneIndex2 = L.boneIndices[4 * i + 2], weight2 = L.boneWeights[4 * i + 2],
                            boneIndex3 = L.boneIndices[4 * i + 3], weight3 = L.boneWeights[4 * i + 3]
                        };
                    }
                    mesh.boneWeights = bw;
                    var skinT = data.skinBones.Select(j => bones[j]).ToArray();
                    mesh.bindposes = skinT.Select(t => t.worldToLocalMatrix * root.transform.localToWorldMatrix).ToArray();
                    var smr = go.AddComponent<SkinnedMeshRenderer>();
                    smr.sharedMesh = mesh; smr.bones = skinT; smr.rootBone = skinT[0];
                    smr.sharedMaterial = mat;
                    mesh.RecalculateBounds();
                    smr.localBounds = new Bounds(mesh.bounds.center, mesh.bounds.size * 1.6f);
                    r = smr;
                }
                else
                {
                    go.AddComponent<MeshFilter>().sharedMesh = mesh;
                    var mr = go.AddComponent<MeshRenderer>();
                    mr.sharedMaterial = mat;
                    mesh.RecalculateBounds();
                    r = mr;
                }
                ctx.AddObjectToAsset("mesh" + k, mesh);
                float h = k < lodScreenHeights.Length ? lodScreenHeights[k] : 0.01f;
                lods.Add(new LOD(h, new[] { r }));
            }
            if (lods.Count > 1)
            {
                var g = root.AddComponent<LODGroup>();
                g.SetLODs(lods.ToArray());
                g.RecalculateBounds();
            }
            if (data.clips != null && data.clips.Length > 0)
            {
                root.AddComponent<Animator>();
                foreach (var c in data.clips)
                {
                    var clip = new AnimationClip { name = c.name, frameRate = 30 };
                    foreach (var cv in c.curves)
                    {
                        string path = PathTo(bones[cv.bone], root.transform);
                        bool rot = cv.property == "rotation";
                        int w = rot ? 4 : 3;
                        var keys = new Keyframe[w][];
                        for (int a = 0; a < w; a++) keys[a] = new Keyframe[cv.times.Length];
                        for (int i = 0; i < cv.times.Length; i++)
                        {
                            for (int a = 0; a < w; a++)
                            {
                                float val = cv.values[w * i + a];
                                if (rot ? (a == 1 || a == 2) : a == 0) val = -val;
                                keys[a][i] = new Keyframe(cv.times[i], val);
                            }
                        }
                        string[] names = rot ? new[] { "m_LocalRotation.x", "m_LocalRotation.y", "m_LocalRotation.z", "m_LocalRotation.w" }
                                             : new[] { "m_LocalPosition.x", "m_LocalPosition.y", "m_LocalPosition.z" };
                        for (int a = 0; a < w; a++)
                        {
                            var curve = new AnimationCurve(keys[a]);
                            for (int i = 0; i < curve.length; i++)
                            {
                                var mode = cv.step ? AnimationUtility.TangentMode.Constant : AnimationUtility.TangentMode.Linear;
                                AnimationUtility.SetKeyLeftTangentMode(curve, i, mode);
                                AnimationUtility.SetKeyRightTangentMode(curve, i, mode);
                            }
                            AnimationUtility.SetEditorCurve(clip, EditorCurveBinding.FloatCurve(path, typeof(Transform), names[a]), curve);
                        }
                    }
                    clip.EnsureQuaternionContinuity();
                    var st = AnimationUtility.GetAnimationClipSettings(clip);
                    st.loopTime = c.loop;
                    AnimationUtility.SetAnimationClipSettings(clip, st);
                    ctx.AddObjectToAsset("clip_" + c.name, clip);
                }
            }
            ctx.AddObjectToAsset("root", root);
            ctx.SetMainObject(root);
        }
    }

    // After import: <name>.controller next to the model with a state per clip (idle first).
    public class PixelPipelineModelPostprocessor : AssetPostprocessor
    {
        static void OnPostprocessAllAssets(string[] imported, string[] deleted, string[] moved, string[] movedFrom)
        {
            foreach (var path in imported)
            {
                if (!path.EndsWith(".ppmodel", StringComparison.OrdinalIgnoreCase)) continue;
                var clips = AssetDatabase.LoadAllAssetsAtPath(path).OfType<AnimationClip>().ToArray();
                if (clips.Length == 0) continue;
                string cpath = Path.ChangeExtension(path, ".controller");
                if (File.Exists(cpath)) continue;
                var ctrl = AnimatorController.CreateAnimatorControllerAtPath(cpath);
                var sm = ctrl.layers[0].stateMachine;
                foreach (var clip in clips.OrderBy(c => c.name == "idle" ? 0 : 1))
                {
                    var s = sm.AddState(clip.name);
                    s.motion = clip;
                    if (clip.name == "idle" || sm.defaultState == null) sm.defaultState = s;
                }
                AssetDatabase.SaveAssets();
            }
        }
    }
}
'''
