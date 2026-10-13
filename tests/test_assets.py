import json

from pixel_pipeline import assets as A


def test_projects_style_and_current(monkeypatch):
    monkeypatch.delenv("PIXEL_LIBRARY", raising=False)
    first = A.current_project()
    assert first.name == "Default" and (first.folder / "style.json").exists()
    retro = A.create_project("Retro", A.Style(palette="pico-8", person_height=32, directions=8))
    assert retro.style.person_frame() == (48, 32) and retro.style.scale(64) == 32
    A.set_current_project("Retro")
    assert A.current_project().name == "Retro"
    assert {p.name for p in A.list_projects()} == {"Default", "Retro"}
    from pixel_pipeline import library
    assert library.root() == retro.folder / "characters"
    assert library.frame_size("person", "normal") == (48, 32)


def test_assets_search_tags_and_collections(monkeypatch):
    monkeypatch.delenv("PIXEL_LIBRARY", raising=False)
    sword = A.new_asset("item", "Iron Sword", prompt="a plain iron sword")
    A.new_asset("item", "Iron Sword")                      # same name: a second folder
    potion = A.new_asset("item", "Red Potion")
    tree = A.new_asset("prop", "Oak Tree")
    A.set_tags(sword, ["weapon", "metal"], ["starter kit"])
    A.set_tags(tree, ["forest"], ["forest set"])
    names = [a.folder.name for a in A.list_assets("item")]
    assert set(names) == {"iron-sword", "iron-sword-2", "red-potion"}
    assert [a.name for a in A.list_assets(query="plain sword")] == ["Iron Sword"]
    assert [a.name for a in A.list_assets(tag="forest")] == ["Oak Tree"]
    assert [a.name for a in A.list_assets(collection="starter kit")] == ["Iron Sword"]
    assert A.all_tags() == ["forest", "metal", "weapon"]
    assert json.loads((potion.folder / "asset.json").read_text())["kind"] == "item"


def test_old_character_library_moves_into_default(monkeypatch, tmp_path):
    monkeypatch.delenv("PIXEL_LIBRARY", raising=False)
    home = tmp_path / "home"
    monkeypatch.setenv("PIXEL_HOME", str(home))
    old = home / "Characters" / "knight"
    old.mkdir(parents=True)
    (old / "character.json").write_text(json.dumps({"name": "Knight", "kind": "person",
                                                    "size": "normal", "role": "knight"}))
    chars = A.list_assets("character")
    assert [c.name for c in chars] == ["Knight"] and chars[0].extra["body"] == "person"
    assert not (home / "Characters").exists()
