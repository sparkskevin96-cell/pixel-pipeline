// Copied into <project>/Assets/Editor/ by verify_uikit.py and run with
//   Unity -batchmode -nographics -projectPath <p> -executeMethod VerifyUIKit.Run -quit
// Reads <project>/expected_uikit.json, checks the imported UI kit (sprites, 9-slice borders,
// the pixel font built from the .fnt, the demo Canvas) and writes <project>/results_uikit.json.
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using UnityEditor;
using UnityEngine;
using UnityEngine.UI;

public static class VerifyUIKit
{
    [Serializable] public class ExpPiece { public string file; public int w, h, left, top, right, bottom; public bool nine; }
    [Serializable] public class Expected { public string kit, font; public int chars; public ExpPiece[] pieces; }
    [Serializable] public class Check { public string name; public bool ok; public string detail; }
    [Serializable] public class Results { public string unity; public bool all_ok; public List<Check> checks = new List<Check>(); }

    static Results res;

    static void Add(string name, bool ok, string detail = "")
    {
        res.checks.Add(new Check { name = name, ok = ok, detail = detail });
    }

    public static void Run()
    {
        string root = Path.GetDirectoryName(Application.dataPath);
        res = new Results { unity = Application.unityVersion };
        try
        {
            AssetDatabase.Refresh(ImportAssetOptions.ForceSynchronousImport);
            var exp = JsonUtility.FromJson<Expected>(File.ReadAllText(Path.Combine(root, "expected_uikit.json")));
            int badBorder = 0, badImport = 0, nine = 0;
            foreach (var p in exp.pieces)
            {
                var imp = AssetImporter.GetAtPath(p.file) as TextureImporter;
                var sprite = AssetDatabase.LoadAssetAtPath<Sprite>(p.file);
                if (imp == null || sprite == null || imp.textureType != TextureImporterType.Sprite
                    || imp.filterMode != FilterMode.Point
                    || imp.textureCompression != TextureImporterCompression.Uncompressed
                    || (int)sprite.rect.width != p.w || (int)sprite.rect.height != p.h)
                {
                    badImport++;
                    Add("import " + p.file, false, imp == null ? "no importer" : imp.textureType + " " + imp.filterMode + " " + (sprite == null ? "no sprite" : sprite.rect.ToString()));
                    continue;
                }
                var want = p.nine ? new Vector4(p.left, p.bottom, p.right, p.top) : Vector4.zero;
                if (p.nine) nine++;
                if (sprite.border != want)
                {
                    badBorder++;
                    Add("border " + p.file, false, sprite.border + " expected " + want);
                }
            }
            Add("pieces import as point-filtered uncompressed sprites", badImport == 0, exp.pieces.Length + " pieces, " + badImport + " bad");
            Add("9-slice borders match kit.json", badBorder == 0 && nine > 10, nine + " sliced, " + badBorder + " bad");

            var font = PixelPipelineUIKit.BuildFont(exp.font);
            Add("font built from .fnt", font != null, exp.font);
            if (font != null)
            {
                Add("font characters", font.characterInfo.Length == exp.chars, font.characterInfo.Length + " vs " + exp.chars);
                Add("font has A and ~", font.HasCharacter('A') && font.HasCharacter('~'));
                var saved = AssetDatabase.LoadAssetAtPath<Font>(Path.ChangeExtension(exp.font, ".fontsettings"));
                Add("font asset saved", saved != null && saved.characterInfo.Length == exp.chars,
                    saved == null ? "missing" : saved.characterInfo.Length.ToString());
                Add("font material uses the atlas", font.material != null && font.material.mainTexture != null);
            }

            var demo = PixelPipelineUIKit.BuildDemo(exp.kit);
            var images = demo.GetComponentsInChildren<Image>(true);
            var sliced = images.Where(i => i.type == Image.Type.Sliced && i.sprite != null && i.hasBorder).ToList();
            var texts = demo.GetComponentsInChildren<Text>(true);
            Add("demo builds", demo != null && images.Length > 40, images.Length + " images");
            Add("demo sliced images", sliced.Count >= 20, sliced.Count.ToString());
            Add("demo images have sprites", images.Count(i => i.sprite == null) == 1, images.Count(i => i.sprite == null) + " without (the backdrop)");
            Add("demo texts use the pixel font", texts.Length > 10 && texts.All(t => t.font == font), texts.Length + " texts");
            var resume = texts.FirstOrDefault(t => t.text == "Resume");
            Add("pixel font lays out text", resume != null && resume.preferredWidth > 10,
                resume == null ? "no Resume" : resume.preferredWidth.ToString());
        }
        catch (Exception e) { Add("exception", false, e.ToString()); }
        res.all_ok = res.checks.Count > 0 && res.checks.All(c => c.ok);
        File.WriteAllText(Path.Combine(root, "results_uikit.json"), JsonUtility.ToJson(res, true));
        EditorApplication.Exit(res.all_ok ? 0 : 1);
    }
}
