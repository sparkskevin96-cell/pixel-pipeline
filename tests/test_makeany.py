"""Make anything: the words router, plans, running with fake pipelines, batches."""
import pytest

from pixel_pipeline import makeany as M

# (words, kind, params that must match)
TABLE = [
    # characters: people and animals
    ("a wolf", "character", {"body": "animal", "drawn": True}),       # animals are drawn too
    ("a knight in a red cape", "character", {"body": "person", "drawn": True, "role": "knight"}),
    ("fire mage", "character", {"body": "person", "role": "mage"}),
    ("goblin", "character", {"body": "person"}),
    ("skeleton warrior", "character", {"body": "person"}),
    ("a big brown bear", "character", {"body": "animal", "size": "large"}),
    ("small fox", "character", {"body": "animal", "size": "small"}),
    ("ice golem", "character", {"body": "person"}),
    ("red dragon", "character", {"body": "animal"}),
    ("old fisherman with a pipe", "character", {"body": "person"}),
    ("orc enemy", "character", {"body": "person"}),
    ("two wolves", "character", {"body": "animal"}),
    ("the village blacksmith", "character", {"body": "person"}),
    ("giant spider", "character", {"body": "animal"}),
    # items
    ("rusty iron sword", "item", {"category": "weapon"}),
    ("sword icon", "item", {"category": "weapon"}),
    ("health potion", "item", {"category": "potion"}),
    ("skeleton key", "item", {"category": "key"}),
    ("golden ring", "item", {"category": "misc"}),
    ("wooden shield", "item", {"category": "armour"}),
    ("a loaf of bread", "item", {"category": "food"}),
    ("magic staff", "item", {"category": "weapon"}),
    ("fire staff", "item", {"category": "weapon"}),
    ("crossbow bolt", "item", {"category": "weapon"}),
    ("goblin's dagger", "item", {"category": "weapon"}),
    ("knight helmet", "item", {"category": "armour"}),
    ("treasure map icon", "item", {}),
    ("ice crystal", "item", {"category": "misc"}),
    # props and buildings
    ("tavern with a thatched roof", "prop", {"building": True, "directions": 4}),
    ("stone wall prop", "prop", {}),
    ("stone wall", "prop", {}),
    ("wooden barrel", "prop", {"height_people": 0.6}),
    ("treasure chest", "prop", {}),
    ("a pine tree", "prop", {"height_people": 2.5}),
    ("horse cart", "prop", {}),
    ("potion shop", "prop", {"building": True}),
    ("card table", "prop", {}),
    ("wizard tower", "prop", {"building": True}),
    ("torch", "prop", {}),
    ("well with a bucket", "prop", {}),
    # tilesets
    ("cobblestone road tiles", "tileset", {"terrain": "road"}),
    ("stone wall tiles", "tileset", {"terrain": "wall"}),
    ("grass tiles", "tileset", {"terrain": "grass"}),
    ("sandy beach", "tileset", {"terrain": "sand"}),
    ("river tiles", "tileset", {"terrain": "water"}),
    ("dungeon floor", "tileset", {"terrain": "stone"}),
    ("snow terrain", "tileset", {"terrain": "snow"}),
    ("rocky cliff", "tileset", {"terrain": "cliff"}),
    ("wooden fence tiles", "tileset", {"terrain": "fence"}),
    ("muddy dirt path", "tileset", {"terrain": "dirt"}),
    # effects
    ("ice nova spell", "effect", {"effect": "frost_nova", "theme": "ice"}),
    ("fire spell", "effect", {"effect": "fire_burst", "theme": "fire"}),
    ("sword slash effect", "effect", {"effect": "sword_combo"}),
    ("sword slash", "effect", {"effect": "sword_combo"}),
    ("fireball", "effect", {"effect": "fireball", "theme": "fire"}),
    ("lightning bolt", "effect", {"effect": "lightning_bolt"}),
    ("chain lightning", "effect", {"effect": "chain_lightning"}),
    ("healing spell", "effect", {"effect": "holy_light"}),
    ("poison cloud", "effect", {"effect": "poison_cloud", "theme": "poison"}),
    ("big explosion", "effect", {"effect": "big_explosion"}),
    ("level up", "effect", {"effect": "level_up_burst"}),
    ("magic circle", "effect", {"effect": "magic_circle"}),
    ("hit sparks", "effect", {"effect": "hit_spark"}),
    ("smoke puff", "effect", {"effect": "smoke"}),
    ("dark magic", "effect", {"effect": "dark_magic", "theme": "dark"}),
    ("green poison spell", "effect", {"effect": "poison_cloud"}),
    ("frost spell", "effect", {"effect": "ice_shard", "theme": "ice"}),
    ("water splash", "effect", {"effect": "splash"}),
    ("lightning", "effect", {"effect": "lightning_bolt"}),
    # portraits
    ("goblin portrait, angry", "portrait", {"who": "goblin", "expressions": "neutral, angry"}),
    ("portrait of a smiling queen", "portrait", {"expressions": "neutral, happy"}),
    ("old wizard face, sad and surprised", "portrait", {"expressions": "neutral, sad, surprised"}),
    ("knight portraits", "portrait", {"who": "knight"}),
    # cards
    ("fire card", "card", {"faction": "fire", "type": "Spell"}),
    ("knight card", "card", {"type": "Unit"}),
    ("legendary dragon card", "card", {"rarity": "legendary", "type": "Unit"}),
    ("healing potion card", "card", {"type": "Item"}),
    ("shadow curse card", "card", {"faction": "shadow"}),
    # UI kits
    ("wooden UI panel", "ui", {"theme": "wood"}),
    ("sci-fi hud", "ui", {"theme": "sci-fi"}),
    ("stone menu buttons", "ui", {"theme": "stone"}),
    ("parchment dialogue box", "ui", {"theme": "parchment"}),
    ("health bar", "ui", {}),
    ("crystal inventory slots", "ui", {"theme": "crystal"}),
    ("gold ui kit", "ui", {"theme": "metal"}),
    # maps
    ("a forest map with a river", "map", {"template": "forest"}),
    ("castle siege map", "map", {"template": "castle"}),
    ("snowy battlefield", "map", {"template": "snowfield"}),
    ("harbour level", "map", {"template": "harbour"}),
    ("village map", "map", {"template": "village"}),
    ("a small arena", "map", {"template": "flat"}),
    # weather / atmosphere
    ("rain at night", "weather", {"weather": "rain", "tod": "night"}),
    ("heavy rain", "weather", {"weather": "rain", "intensity": 1.6}),
    ("snow", "weather", {"weather": "snow", "tod": ""}),
    ("light snow at dawn", "weather", {"weather": "snow", "tod": "dawn", "intensity": 0.5}),
    ("fog", "weather", {"weather": "fog"}),
    ("misty morning", "weather", {"weather": "fog", "tod": "dawn"}),
    ("fireflies", "weather", {"weather": "fireflies"}),
    ("god rays", "weather", {"weather": "god_rays"}),
    ("light shafts in a forest", "weather", {"weather": "god_rays"}),
    ("falling leaves", "weather", {"weather": "leaves"}),
    ("cherry blossoms", "weather", {"weather": "petals"}),
    ("sandstorm", "weather", {"weather": "sandstorm"}),
    ("rising embers", "weather", {"weather": "embers"}),
    ("cloud shadows", "weather", {"weather": "cloud_shadows"}),
    ("dust motes", "weather", {"weather": "dust"}),
    # backgrounds / skies
    ("sunset mountains background", "background", {"scene": "mountains", "tod": "sunset", "words": ""}),
    ("parallax forest", "background", {"scene": "forest", "words": ""}),
    ("night sky", "background", {"tod": "night", "words": ""}),
    ("starry night sky", "background", {"tod": "night", "words": ""}),
    ("desert background at noon", "background", {"scene": "desert", "tod": "noon"}),
    ("misty bamboo valley background", "background", {"words": "misty bamboo valley"}),
    # animated water
    ("animated lagoon water", "water", {"theme": "lagoon"}),
    ("ocean waves", "water", {"theme": "ocean"}),
    ("flowing river", "water", {"theme": "river"}),
    ("animated lava", "water", {"theme": "lava"}),
]


def test_router_table_is_big_and_covers_every_kind():
    assert len(TABLE) >= 60
    assert {k for _, k, _ in TABLE} == set(M.KINDS)


@pytest.mark.parametrize("words,kind,params", TABLE, ids=[t[0] for t in TABLE])
def test_router_classifies(words, kind, params):
    guesses = M.classify(words)
    best = guesses[0]
    assert best.kind == kind, f"{words!r} -> {[(g.kind, g.score) for g in guesses]}"
    for k, v in params.items():
        assert best.plan.params.get(k) == v, (words, k, best.plan.params)


def test_router_accuracy_report():
    right = sum(M.classify(w)[0].kind == k for w, k, _ in TABLE)
    assert right == len(TABLE)


def test_tricky_pairs_go_different_ways():
    pick = lambda w: M.classify(w)[0].kind                         # noqa: E731
    assert (pick("sword icon"), pick("sword slash effect")) == ("item", "effect")
    assert (pick("stone wall tiles"), pick("stone wall prop")) == ("tileset", "prop")
    assert (pick("fire mage"), pick("fire spell"), pick("fire card")) == ("character", "effect", "card")
    assert (pick("skeleton key"), pick("skeleton archer")) == ("item", "character")


def test_alternatives_offer_other_kinds():
    guesses = M.classify("stone wall")
    kinds = [g.kind for g in guesses]
    assert kinds[0] == "prop" and "tileset" in kinds[1:] and len(set(kinds)) == len(kinds)
    assert all(g.plan.kind == g.kind for g in guesses)
    tiles = next(g for g in guesses if g.kind == "tileset")
    assert tiles.plan.params["terrain"] == "wall"
    fire_mage = [g.kind for g in M.classify("fire mage")]
    assert fire_mage[0] == "character" and "effect" in fire_mage
    unknown = M.classify("a zorblax")
    assert [g.kind for g in unknown] == list(M.FALLBACK) and "guess" in unknown[0].plan.why
    assert M.classify("   ") == []


def test_names_prompts_and_summaries():
    p = M.route("tavern with a thatched roof")
    assert p.name == "Tavern" and p.prompt == "tavern with a thatched roof"
    assert M.route("sword icon").prompt == "sword" and M.route("sword icon").name == "Sword"
    assert M.route("cobblestone road tiles").name == "Cobblestone road tiles"
    assert M.route("wooden UI panel").name == "Wood UI kit"
    assert M.route("a forest map with a river").name == "Forest map"
    assert M.route("goblin portrait, angry").name == "Goblin portraits"
    wolf = M.route("a wolf")
    text = wolf.summary()
    assert text.startswith("I'll make: Character (animal) “Wolf”") and "16 directions" in text
    assert "drawn sprites" in text and "min of GPU" in text
    knight = M.route("a knight")
    assert "drawn sprites" in knight.summary() and knight.gpu_minutes() == wolf.gpu_minutes()
    undrawn = M.Plan("character", "Wolf", "a wolf", {**wolf.params, "drawn": False})
    assert "3D-rendered" in undrawn.summary() and undrawn.gpu_minutes() < wolf.gpu_minutes()
    assert "no GPU" in M.route("rain at night").summary()
    assert "no GPU" in M.route("animated lagoon water").summary()
    assert "no GPU" in M.route("sunset mountains background").summary()
    assert "GPU" in M.route("misty bamboo valley background").summary()
    assert M.route("rain at night").name == "Rain (night)"
    assert M.route("night sky").name == "Night sky background"
    assert "no GPU" in M.route("wooden UI panel").summary()
    assert "no GPU" in M.route("ice nova spell").summary()
    assert "not in your library" in M.route("goblin portrait").summary()
    assert M.route("rusty iron sword").gpu_minutes() < 3
    for kind in M.KINDS:                        # every kind has editable fields its plan fills
        plan = M.plan_for(kind, "a thing")
        for key, _label, _spec in M.fields_for(kind):
            assert key in plan.params, (kind, key)
        assert M.Plan.from_dict(plan.to_dict()) == plan


def test_plan_follows_the_project_style():
    from pixel_pipeline.assets import Style
    side = Style(directions=2)
    assert M.plan_for("character", "a knight", style=side).params["directions"] == 2
    assert M.plan_for("prop", "a house", style=side).params["directions"] == 1
    assert M.plan_for("prop", "a house", style=Style()).params["directions"] == 4


def test_llm_hook_is_used_then_falls_back():
    try:
        M.set_llm_router(lambda text: [("card", {"faction": "water"}), ("nonsense", {})])
        g = M.classify("a wolf")
        assert g[0].kind == "card" and g[0].plan.params["faction"] == "water" and len(g) == 1
        M.set_llm_router(lambda text: (_ for _ in ()).throw(RuntimeError("offline")))
        assert M.classify("a wolf")[0].kind == "character"
        assert M.classify("a wolf", use_llm=False)[0].kind == "character"
    finally:
        M.set_llm_router(None)


def test_parse_batch():
    text = "a wolf\n\n# comment\n- rusty iron sword\n2) ice nova spell\n  • fire card  \n"
    assert M.parse_batch(text) == ["a wolf", "rusty iron sword", "ice nova spell", "fire card"]


# --------------------------------------------------------------------------- running (no GPU)

def fake_makers(calls):
    """Pipelines that save a real asset but never touch the GPU."""
    from pixel_pipeline import assets

    def maker(kind):
        def make(plan, project, say, client=None):
            calls.append(plan.kind)
            say(f"faking {plan.name}")
            if plan.kind == "tileset":
                import numpy as np
                return M.Made(plan, None, [np.zeros((8, 8, 4), np.uint8)] * 2, 1, {"swatches": []})
            a = assets.new_asset(kind if kind in assets.KINDS else "prop", plan.name, project,
                                 prompt=plan.prompt)
            return M.Made(plan, a)
        return make
    return {k: maker(k) for k in M.KINDS}


def test_run_plan_uses_the_kinds_maker(tmp_path):
    calls = []
    from pixel_pipeline.assets import create_project
    proj = create_project("MakeTest")
    made = M.run_plan(M.route("rusty iron sword"), proj, makers=fake_makers(calls))
    assert calls == ["item"] and made.saved and made.folder.exists() and made.name == "Rusty iron sword"
    with pytest.raises(ValueError):
        M.plan_for("spaceship", "x")


def test_real_procedural_pipelines_make_assets(tmp_path):
    """UI kits, maps, effects, weather, water and procedural backgrounds need no GPU: run them
    for real."""
    from pixel_pipeline.assets import create_project, list_assets
    proj = create_project("Procedural")
    for words, kind in (("wooden UI panel", "ui"), ("a forest map with a river", "map"),
                        ("ice nova spell", "effect"), ("rain at night", "weather"),
                        ("animated lagoon water", "water"),
                        ("sunset mountains background", "background")):
        made = M.run_plan(M.route(words), proj)
        assert made.saved and made.folder.exists(), words
        assert [a.name for a in list_assets(kind, project=proj)] == [made.name]
    fx = list_assets("effect", project=proj)[0]
    assert fx.extra["effect"] == "frost_nova" and fx.extra["theme"] == "ice"
    rain = list_assets("weather", project=proj)[0]
    assert rain.extra["weather"] == "rain" and rain.extra["tod"] == "night"
    assert list_assets("water", project=proj)[0].extra["theme"] == "lagoon"
    bg = list_assets("background", project=proj)[0]
    assert bg.extra["scene"] == "mountains" and bg.extra["tod"] == "sunset" and not bg.extra.get("prompt")


def test_keep_picks_variations(tmp_path, monkeypatch):
    from pixel_pipeline import objects
    from pixel_pipeline.assets import create_project, new_asset
    proj = create_project("Keep")
    picked = []
    monkeypatch.setattr(objects, "choose_icon", lambda a, i, p=None: picked.append(("icon", i)))
    asset = new_asset("item", "Sword", proj)
    made = M.Made(M.route("sword icon"), asset, ["a", "b", "c"], 1, {"project": proj})
    M.keep(made, 1)
    assert picked == []                         # already the chosen one
    M.keep(made, 3)
    assert picked == [("icon", 3)] and made.chosen == 3


def test_batch_goes_through_the_paced_queue(tmp_path):
    from pixel_pipeline import jobqueue
    from pixel_pipeline.assets import create_project
    proj = create_project("Batch")
    calls = []
    makers = fake_makers(calls)
    q = jobqueue.Queue(proj)
    other = q.add("item", "Not mine", "x")                       # someone else's job waits
    plans = [M.route(t) for t in M.parse_batch("a wolf\nrusty iron sword\ncobblestone road tiles")]
    jobs = M.queue_plans(proj, plans, queue=q)
    assert [j.kind for j in jobs] == ["make"] * 3
    slept = []

    def worker(name, prompt, options, project):
        if options["plan"]["kind"] == "tileset":
            raise RuntimeError("no GPU here")
        return M.queue_worker(name, prompt, options, project, makers=makers)
    done = jobqueue.run_queue(proj, {"make": worker}, pause_s=0.5, queue=q, sleep=slept.append,
                              job_ids={j.id for j in jobs})
    assert [j.name for j in done] == ["Wolf", "Rusty iron sword", "Cobblestone road tiles"]
    assert calls == ["character", "item"] and q.get(other.id).status == "pending"
    assert slept                                                  # rested between jobs
    text = M.batch_summary(done)
    assert text.splitlines()[0] == "2 made, 1 failed" and "no GPU here" in text
    assert "make" in jobqueue.default_workers()


# --------------------------------------------------------------------------- the real makers, fake ComfyUI

from test_generate import comfy, white_key  # noqa: E402,F401  (pytest fixtures)


def test_real_makers_against_the_fake_comfyui(comfy, monkeypatch):
    """Item, drawn prop, card, tileset and words-only portraits through the real pipelines;
    the GPU is the fake ComfyUI from test_generate."""
    pytest.importorskip("trimesh")
    from pixel_pipeline import teams
    from pixel_pipeline.assets import create_project, list_assets
    monkeypatch.setattr(teams, "_face_model_path", None)          # no YuNet download
    monkeypatch.delenv("PIXEL_LIBRARY", raising=False)
    fake, client = comfy
    proj = create_project("RealMakers")
    sword = M.run_plan(M.route("rusty iron sword"), proj, client=client)
    assert sword.saved and len(sword.variations) == 3 and sword.asset.extra["category"] == "weapon"
    M.keep(sword, 2)
    assert sword.asset.extra["icon_choice"] == 2
    barrel = M.run_plan(M.route("wooden barrel"), proj, client=client)
    assert barrel.saved and barrel.asset.extra.get("look") == "drawn" and len(barrel.variations) == 3
    M.keep(barrel, 3)
    assert barrel.asset.extra["chosen"] == 3
    card = M.run_plan(M.route("fire card"), proj, client=client)
    assert card.saved and len(card.variations) == 3 and card.asset.extra["card"]["faction"] == "fire"
    tiles = M.run_plan(M.route("cobblestone road tiles"), proj, client=client)
    assert not tiles.saved and len(tiles.variations) == 3
    M.keep(tiles, 2)
    assert tiles.saved and tiles.asset.extra["terrain"] == "road"
    faces = M.run_plan(M.route("goblin portrait, angry"), proj, client=client)
    assert faces.saved and faces.asset.extra.get("expressions", [])[:2] == ["neutral", "angry"]
    assert (proj.folder / "portrait_subjects" / "goblin" / "picture.png").exists()
    kinds = {a.kind for a in list_assets(project=proj)}
    assert kinds >= {"item", "prop", "card", "tileset", "portrait"} and "character" not in kinds


def test_character_maker_passes_the_plan(monkeypatch, tmp_path):
    from pixel_pipeline import library, model3d
    from pixel_pipeline.assets import create_project
    seen = {}

    def fake_model(prompt, out, subject="person", **kw):
        out.write_bytes(b"glb")
        seen["model"] = (prompt, subject)
        return model3d.ModelResult(out, None, None, 1)

    def fake_add(name, model, **kw):
        seen["add"] = (name, kw["kind"], kw["size"], kw["drawn"], kw["prompt"], kw["role"])
        return library.Character(name, kw["kind"], kw["size"] or "normal", kw["role"], 1.0, tmp_path / name)
    monkeypatch.setattr(model3d, "prompt_to_model", fake_model)
    monkeypatch.setattr(library, "add_character", fake_add)
    proj = create_project("Chars")
    M.run_plan(M.route("a knight in a red cape"), proj)
    assert seen["model"] == ("a knight in a red cape", "person")
    assert seen["add"] == ("Knight", "person", "normal", True, "a knight in a red cape", "knight")
    M.run_plan(M.route("a big brown bear"), proj)
    assert seen["add"][1:4] == ("animal", "large", True)      # drawn (3D render without ComfyUI)
