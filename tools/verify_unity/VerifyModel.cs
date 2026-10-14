// Copied into <project>/Assets/Editor/ by verify_model.py and run with
//   Unity -batchmode -nographics -projectPath <p> -executeMethod VerifyModel.Run -quit
// Reads <project>/expected_model.json, checks each imported .ppmodel prefab (LODs, triangle
// counts, materials + textures, bones, clips, that a clip moves the skinned mesh, the
// controller) and writes <project>/results_model.json.
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using UnityEditor;
using UnityEditor.Animations;
using UnityEngine;

public static class VerifyModel
{
    [Serializable] public class ExpModel { public string path; public bool skinned; public int bones; public string[] clips; public int[] tris; public string filter; }
    [Serializable] public class Expected { public ExpModel[] models; }
    [Serializable] public class Check { public string name; public bool ok; public string detail; }
    [Serializable] public class Results { public string unity; public bool all_ok; public List<Check> checks = new List<Check>(); }
    static Results res;
    static void Add(string name, bool ok, string detail = "") { res.checks.Add(new Check { name = name, ok = ok, detail = detail }); }

    public static void Run()
    {
        string root = Path.GetDirectoryName(Application.dataPath);
        res = new Results { unity = Application.unityVersion };
        try
        {
            AssetDatabase.Refresh(ImportAssetOptions.ForceSynchronousImport);
            var exp = JsonUtility.FromJson<Expected>(File.ReadAllText(Path.Combine(root, "expected_model.json")));
            foreach (var m in exp.models)
            {
                string tag = Path.GetFileNameWithoutExtension(m.path) + ": ";
                var prefab = AssetDatabase.LoadAssetAtPath<GameObject>(m.path);
                Add(tag + "imports as a prefab", prefab != null, m.path);
                if (prefab == null) continue;
                var lodg = prefab.GetComponent<LODGroup>();
                var lods = lodg != null ? lodg.GetLODs() : new LOD[0];
                Add(tag + "LODGroup with " + m.tris.Length + " LODs", lods.Length == m.tris.Length, lods.Length + " LODs");
                var meshes = new List<Mesh>();
                foreach (var l in lods)
                {
                    var r = l.renderers[0];
                    var smr0 = r as SkinnedMeshRenderer;
                    meshes.Add(smr0 != null ? smr0.sharedMesh : r.GetComponent<MeshFilter>().sharedMesh);
                }
                var counts = meshes.Select(x => x.triangles.Length / 3).ToArray();
                Add(tag + "triangle counts", counts.SequenceEqual(m.tris), string.Join(",", counts));
                var r0 = lods.Length > 0 ? lods[0].renderers[0] : null;
                var mat = r0 != null ? r0.sharedMaterial : null;
                Texture tex = null;
                if (mat != null)
                    tex = mat.HasProperty("_BaseMap") && mat.GetTexture("_BaseMap") != null ? mat.GetTexture("_BaseMap") : mat.GetTexture("_MainTex");
                Add(tag + "material has the albedo texture", tex != null,
                    mat == null ? "no material" : mat.shader.name + " " + (tex == null ? "no tex" : tex.width + "px " + tex.filterMode));
                if (tex != null)
                    Add(tag + "texture filter " + m.filter, (m.filter == "point") == (tex.filterMode == FilterMode.Point), tex.filterMode.ToString());
                if (!m.skinned) continue;
                var smr = r0 as SkinnedMeshRenderer;
                Add(tag + "skinned with " + m.bones + " bones",
                    smr != null && smr.bones.Length == m.bones && meshes[0].bindposes.Length == m.bones,
                    smr == null ? "not skinned" : smr.bones.Length + " bones");
                var clips = AssetDatabase.LoadAllAssetsAtPath(m.path).OfType<AnimationClip>().ToArray();
                var names = clips.Select(c => c.name).OrderBy(x => x).ToArray();
                Add(tag + "clips", m.clips.OrderBy(x => x).SequenceEqual(names), string.Join(",", names));
                var walk = clips.FirstOrDefault(c => c.name == "walk") ?? clips.FirstOrDefault();
                if (walk != null && smr != null)
                {
                    var inst = (GameObject)UnityEngine.Object.Instantiate(prefab);
                    var ismr = inst.GetComponentsInChildren<SkinnedMeshRenderer>()[0];
                    var a = new Mesh();
                    var b = new Mesh();
                    walk.SampleAnimation(inst, 0f);
                    ismr.BakeMesh(a);
                    walk.SampleAnimation(inst, walk.length * 0.5f);
                    ismr.BakeMesh(b);
                    float moved = 0;
                    var va = a.vertices;
                    var vb = b.vertices;
                    for (int i = 0; i < va.Length; i++) moved = Mathf.Max(moved, (va[i] - vb[i]).magnitude);
                    Add(tag + "clip " + walk.name + " moves the skinned mesh", moved > 0.02f,
                        "max vertex move " + moved.ToString("F3") + ", posed size " + a.bounds.size);
                    var rest = meshes[0].bounds.size;
                    Add(tag + "posed mesh keeps its size (bind poses right)",
                        Mathf.Abs(a.bounds.size.y - rest.y) < 0.25f * rest.y,
                        "rest " + rest + " posed " + a.bounds.size);
                    var st = AnimationUtility.GetAnimationClipSettings(walk);
                    Add(tag + "clip " + walk.name + " loops", st.loopTime, st.loopTime.ToString());
                    UnityEngine.Object.DestroyImmediate(inst);
                }
                var ctrl = AssetDatabase.LoadAssetAtPath<AnimatorController>(Path.ChangeExtension(m.path, ".controller"));
                Add(tag + "controller with a state per clip",
                    ctrl != null && ctrl.layers[0].stateMachine.states.Length == clips.Length,
                    ctrl == null ? "none" : ctrl.layers[0].stateMachine.states.Length + " states, default " + ctrl.layers[0].stateMachine.defaultState.name);
            }
        }
        catch (Exception e) { Add("exception", false, e.ToString()); }
        res.all_ok = res.checks.Count > 0 && res.checks.All(c => c.ok);
        File.WriteAllText(Path.Combine(root, "results_model.json"), JsonUtility.ToJson(res, true));
    }
}
