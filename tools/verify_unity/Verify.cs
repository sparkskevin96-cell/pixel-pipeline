// Copied into <project>/Assets/Editor/ by verify_unity.py and run with
//   Unity -batchmode -nographics -projectPath <p> -executeMethod Verify.Run -quit
// Reads <project>/expected.json, checks how Unity imported each exported asset and writes
// <project>/results.json. Every check is {asset, name, ok, detail}.
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using UnityEditor;
using UnityEngine;

public static class Verify
{
    [Serializable] public class ExpFrame { public string name; public int x, y, w, h; public string crop; }
    [Serializable] public class ExpClip { public string path; public string[] sprites; public float[] times; public float length; public float fps; public bool loop; }
    [Serializable] public class ExpAsset
    {
        public string name; public string png; public string normal; public int width, height;
        public float ppu; public bool multiple; public ExpFrame[] frames; public ExpClip[] clips; public string[] singles;
    }
    [Serializable] public class Expected { public ExpAsset[] assets; }
    [Serializable] public class Check { public string asset, name; public bool ok; public string detail; }
    [Serializable] public class Results { public string unity; public bool all_ok; public List<Check> checks = new List<Check>(); }

    static Results res;

    static void Add(string asset, string name, bool ok, string detail = "")
    {
        res.checks.Add(new Check { asset = asset, name = name, ok = ok, detail = detail });
    }

    public static void Run()
    {
        string root = Path.GetDirectoryName(Application.dataPath);
        res = new Results { unity = Application.unityVersion };
        try
        {
            AssetDatabase.Refresh(ImportAssetOptions.ForceSynchronousImport);
            var exp = JsonUtility.FromJson<Expected>(File.ReadAllText(Path.Combine(root, "expected.json")));
            foreach (var a in exp.assets) CheckAsset(root, a);
        }
        catch (Exception e) { Add("-", "exception", false, e.ToString()); }
        res.all_ok = res.checks.Count > 0 && res.checks.All(c => c.ok);
        File.WriteAllText(Path.Combine(root, "results.json"), JsonUtility.ToJson(res, true));
        EditorApplication.Exit(res.all_ok ? 0 : 1);
    }

    static void CheckAsset(string root, ExpAsset a)
    {
        var imp = AssetImporter.GetAtPath(a.png) as TextureImporter;
        Add(a.name, "importer", imp != null, a.png);
        if (imp == null) return;
        Add(a.name, "textureType Sprite", imp.textureType == TextureImporterType.Sprite, imp.textureType.ToString());
        var mode = a.multiple ? SpriteImportMode.Multiple : SpriteImportMode.Single;
        Add(a.name, "spriteImportMode " + mode, imp.spriteImportMode == mode, imp.spriteImportMode.ToString());
        Add(a.name, "filterMode Point", imp.filterMode == FilterMode.Point, imp.filterMode.ToString());
        Add(a.name, "compression none", imp.textureCompression == TextureImporterCompression.Uncompressed,
            imp.textureCompression.ToString());
        var st = imp.GetPlatformTextureSettings("Standalone");
        Add(a.name, "Standalone not compressed", !st.overridden || st.textureCompression == TextureImporterCompression.Uncompressed,
            st.overridden + " " + st.textureCompression);
        Add(a.name, "no mipmaps", !imp.mipmapEnabled, imp.mipmapEnabled.ToString());
        Add(a.name, "pixels per unit", Mathf.Approximately(imp.spritePixelsPerUnit, a.ppu), imp.spritePixelsPerUnit.ToString());

        var tex = AssetDatabase.LoadAssetAtPath<Texture2D>(a.png);
        Add(a.name, "texture loads", tex != null);
        if (tex == null) return;
        Add(a.name, "texture not resized", tex.width == a.width && tex.height == a.height,
            tex.width + "x" + tex.height + " expected " + a.width + "x" + a.height);
        Add(a.name, "texture format uncompressed", tex.format == TextureFormat.RGBA32 || tex.format == TextureFormat.ARGB32
            || tex.format == TextureFormat.RGB24, tex.format.ToString());
        Add(a.name, "texture filter Point", tex.filterMode == FilterMode.Point, tex.filterMode.ToString());

        var sprites = AssetDatabase.LoadAllAssetsAtPath(a.png).OfType<Sprite>().ToList();
        Add(a.name, "sprite count == frames", sprites.Count == a.frames.Length, sprites.Count + " vs " + a.frames.Length);
        var names = sprites.Select(s => s.name).ToList();
        Add(a.name, "sprite names unique", names.Distinct().Count() == names.Count);

        // the PNG as Unity reads pixels (rows bottom-up, the same convention as sprite rects)
        var full = new Texture2D(2, 2, TextureFormat.RGBA32, false);
        full.LoadImage(File.ReadAllBytes(Path.Combine(root, a.png)));
        int missing = 0, badRect = 0, badPivot = 0, badPixels = 0, badTex = 0;
        string firstBad = "";
        var byName = new Dictionary<string, Sprite>();
        foreach (var s in sprites) byName[s.name] = s;
        foreach (var f in a.frames)
        {
            if (!byName.TryGetValue(f.name, out var s)) { missing++; if (firstBad == "") firstBad = "missing " + f.name; continue; }
            if (s.texture != tex) badTex++;
            var r = s.rect;
            int ux = f.x, uy = a.height - f.y - f.h;           // PNG top-left -> Unity bottom-left
            if ((int)r.x != ux || (int)r.y != uy || (int)r.width != f.w || (int)r.height != f.h)
            {
                badRect++;
                if (firstBad == "") firstBad = f.name + " rect " + r + " expected " + ux + "," + uy + "," + f.w + "," + f.h;
            }
            if (Mathf.Abs(s.pivot.x - f.w / 2f) > 0.01f || Mathf.Abs(s.pivot.y) > 0.01f)
            {
                badPivot++;
                if (firstBad == "") firstBad = f.name + " pivot " + s.pivot;
            }
            if (!string.IsNullOrEmpty(f.crop))
            {
                var crop = new Texture2D(2, 2, TextureFormat.RGBA32, false);
                crop.LoadImage(File.ReadAllBytes(Path.Combine(root, f.crop)));
                var want = crop.GetPixels32();
                var got = full.GetPixels((int)r.x, (int)r.y, (int)r.width, (int)r.height);
                bool same = want.Length == got.Length;
                for (int i = 0; same && i < want.Length; i++)
                {
                    Color32 g = got[i];
                    if (want[i].a != g.a || (want[i].a > 0 && (want[i].r != g.r || want[i].g != g.g || want[i].b != g.b))) same = false;
                }
                if (!same) { badPixels++; if (firstBad == "") firstBad = f.name + " pixels differ from the expected crop"; }
                UnityEngine.Object.DestroyImmediate(crop);
            }
        }
        Add(a.name, "every frame has its sprite (by name)", missing == 0, missing + " missing");
        Add(a.name, "sprite rects (y flipped)", badRect == 0, badRect + " wrong. " + firstBad);
        Add(a.name, "pivot bottom-centre", badPivot == 0, badPivot + " wrong");
        Add(a.name, "sprite pixels == PNG crop", badPixels == 0, badPixels + " differ (" + a.frames.Count(f => !string.IsNullOrEmpty(f.crop)) + " compared)");
        Add(a.name, "sprites use the sheet texture", badTex == 0, badTex + " wrong");
        UnityEngine.Object.DestroyImmediate(full);

        if (!string.IsNullOrEmpty(a.normal))
        {
            var nimp = AssetImporter.GetAtPath(a.normal) as TextureImporter;
            Add(a.name, "normal map importer", nimp != null, a.normal);
            if (nimp != null)
            {
                Add(a.name, "normal map type NormalMap", nimp.textureType == TextureImporterType.NormalMap, nimp.textureType.ToString());
                Add(a.name, "normal map filterMode Point", nimp.filterMode == FilterMode.Point, nimp.filterMode.ToString());
                Add(a.name, "normal map no mipmaps", !nimp.mipmapEnabled);
                Add(a.name, "normal map not from greyscale", !nimp.convertToNormalmap);
                var ntex = AssetDatabase.LoadAssetAtPath<Texture2D>(a.normal);
                Add(a.name, "normal map not resized", ntex != null && ntex.width == a.width && ntex.height == a.height,
                    ntex == null ? "null" : ntex.width + "x" + ntex.height);
                // secondary texture on the sheet's sprites (_NormalMap for URP 2D lights)
                if (sprites.Count > 0)
                {
                    var sec = new SecondarySpriteTexture[sprites[0].GetSecondaryTextureCount()];
                    sprites[0].GetSecondaryTextures(sec);
                    bool linked = sec.Any(t => t.name == "_NormalMap" && t.texture == ntex);
                    Add(a.name, "sprites carry _NormalMap secondary texture", linked,
                        string.Join(",", sec.Select(t => t.name + "=" + (t.texture ? t.texture.name : "null"))));
                }
            }
        }

        foreach (var single in a.singles ?? new string[0])
        {
            var simp = AssetImporter.GetAtPath(single) as TextureImporter;
            var ss = AssetDatabase.LoadAllAssetsAtPath(single).OfType<Sprite>().ToList();
            Add(a.name, Path.GetFileName(single) + " imported as one Point sprite",
                simp != null && simp.textureType == TextureImporterType.Sprite && simp.spriteImportMode == SpriteImportMode.Single
                && simp.filterMode == FilterMode.Point && simp.textureCompression == TextureImporterCompression.Uncompressed
                && ss.Count == 1 && Mathf.Abs(ss[0].pivot.x - ss[0].rect.width / 2) < 0.01f && Mathf.Abs(ss[0].pivot.y) < 0.01f,
                simp == null ? "no importer (meta not parsed?)" : simp.spriteImportMode + " " + simp.filterMode + " sprites=" + ss.Count);
        }

        if (a.clips != null)
        {
            int bad = 0, sampledExact = 0; string why = "";
            foreach (var c in a.clips)
            {
                var clip = AssetDatabase.LoadAssetAtPath<AnimationClip>(c.path);
                if (clip == null) { bad++; if (why == "") why = "no clip " + c.path; continue; }
                var bindings = AnimationUtility.GetObjectReferenceCurveBindings(clip);
                var b = bindings.FirstOrDefault(x => x.type == typeof(SpriteRenderer) && x.propertyName == "m_Sprite");
                if (bindings.Length != 1 || b.propertyName != "m_Sprite") { bad++; if (why == "") why = c.path + " bindings " + bindings.Length; continue; }
                var keys = AnimationUtility.GetObjectReferenceCurve(clip, b);
                var got = keys.Select(k => k.value ? k.value.name : "null").ToList();
                // a looping clip may repeat the last sprite as an end key
                var want = c.sprites.ToList();
                bool order = got.Count >= want.Count && got.Take(want.Count).SequenceEqual(want)
                             && got.Skip(want.Count).All(n => n == want[want.Count - 1]);
                bool sameSheet = keys.All(k => k.value is Sprite sp && sp.texture == tex);
                bool times = true;
                for (int i = 0; i < want.Count; i++) if (Mathf.Abs(keys[i].time - c.times[i]) > 0.0011f) times = false;
                bool length = Mathf.Abs(clip.length - c.length) < 0.0011f;
                // what a SpriteRenderer actually shows when the clip is sampled mid-frame;
                // the same for a control clip built through Unity's API from the same sprites
                var sampledNames = SampleNames(clip, c);
                var control = new AnimationClip { frameRate = c.fps };
                AnimationUtility.SetObjectReferenceCurve(control, b, keys);
                var controlNames = SampleNames(control, c);
                UnityEngine.Object.DestroyImmediate(control);
                bool sampled = sampledNames.SequenceEqual(controlNames);
                if (sampledNames.SequenceEqual(want)) sampledExact++;
                var settings = AnimationUtility.GetAnimationClipSettings(clip);
                bool rate = Mathf.Abs(clip.frameRate - c.fps) < 0.001f;
                if (!order || !sameSheet || !times || !length || !sampled || settings.loopTime != c.loop || !rate)
                {
                    bad++;
                    if (why == "") why = c.path + " order=" + order + " sheet=" + sameSheet + " times=" + times + " length=" + clip.length + " sampled=" + string.Join(",", sampledNames.Take(4)) + " control=" + string.Join(",", controlNames.Take(4))
                                         + " loop=" + settings.loopTime + " rate=" + clip.frameRate + " got " + string.Join(",", got.Take(4));
                }
            }
            Add(a.name, "animation clips play the frames in order", bad == 0, a.clips.Length + " clips, " + bad + " bad. " + why);
            Add(a.name, "sampled clips show each frame mid-frame", sampledExact == a.clips.Length,
                sampledExact + "/" + a.clips.Length + " (SpriteRenderer.sprite after AnimationMode.SampleAnimationClip)");
        }
    }
    static List<string> SampleNames(AnimationClip clip, ExpClip c)
    {
        var go = new GameObject("probe");
        var sr = go.AddComponent<SpriteRenderer>();
        go.AddComponent<Animator>();
        var names = new List<string>();
        AnimationMode.StartAnimationMode();
        for (int i = 0; i < c.sprites.Length; i++)
        {
            float t = c.times[i] + 0.5f * ((i + 1 < c.times.Length ? c.times[i + 1] : c.length) - c.times[i]);
            AnimationMode.BeginSampling();
            AnimationMode.SampleAnimationClip(go, clip, t);
            AnimationMode.EndSampling();
            names.Add(sr.sprite ? sr.sprite.name : "null");
        }
        AnimationMode.StopAnimationMode();
        UnityEngine.Object.DestroyImmediate(go);
        return names;
    }

    // Unity's own serialisation, to diff against what exports.py writes: re-serialises the
    // exported .meta files in place and saves a clip built through the API as Assets/PPRef/ref.anim.
    public static void Reference()
    {
        string root = Path.GetDirectoryName(Application.dataPath);
        AssetDatabase.Refresh(ImportAssetOptions.ForceSynchronousImport);
        var exp = JsonUtility.FromJson<Expected>(File.ReadAllText(Path.Combine(root, "expected.json")));
        var paths = new List<string>();
        foreach (var a in exp.assets) { paths.Add(a.png); if (!string.IsNullOrEmpty(a.normal)) paths.Add(a.normal); }
        AssetDatabase.ForceReserializeAssets(paths, ForceReserializeAssetsOptions.ReserializeMetadata);
        var sprites = AssetDatabase.LoadAllAssetsAtPath(exp.assets[0].png).OfType<Sprite>()
            .OrderBy(s => s.rect.y * -100000 + s.rect.x).Take(3).ToArray();
        var clip = new AnimationClip { frameRate = 10 };
        var b = EditorCurveBinding.PPtrCurve("", typeof(SpriteRenderer), "m_Sprite");
        var keys = sprites.Select((sp, i) => new ObjectReferenceKeyframe { time = i / 10f, value = sp }).ToArray();
        AnimationUtility.SetObjectReferenceCurve(clip, b, keys);
        var st = AnimationUtility.GetAnimationClipSettings(clip);
        st.loopTime = true;
        AnimationUtility.SetAnimationClipSettings(clip, st);
        Directory.CreateDirectory(Path.Combine(Application.dataPath, "PPRef"));
        AssetDatabase.CreateAsset(clip, "Assets/PPRef/ref.anim");
        AssetDatabase.SaveAssets();
        EditorApplication.Exit(0);
    }
}
