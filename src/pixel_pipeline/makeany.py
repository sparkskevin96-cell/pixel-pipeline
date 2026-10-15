"""Make anything from words: "a wolf", "rusty iron sword", "ice nova spell" -> the right
pipeline with the right settings.

    guesses = classify("goblin portrait, angry")   # best first, then "it could also be…"
    plan = guesses[0].plan                          # Plan(kind="portrait", params={...})
    plan.summary()                                  # "I'll make: Portraits of 'Goblin'…"
    made = run_plan(plan, progress=print)           # saved in the project (GPU where needed)
    keep(made, 2)                                   # pick variation 2 (icons, drawings, art…)

The router is rule-based (keyword cues + small lexicons + "the last noun is the head"), so
it is deterministic, testable and needs no network. `set_llm_router(fn)` is the hook for a
smarter router (e.g. Claude with an Anthropic key from keys.json); nothing here makes network
calls, and the rules are always the fallback.

Kinds: character (person | animal), item, prop (props and buildings), tileset (terrain),
effect (kind + theme), portrait (expressions), card, ui (UI kit), map (template), weather
(weather.py: rain, snow, fog, god rays, fireflies…), water (water.py: animated water + shores),
background (backgrounds.py: parallax skies and scenery; from words on the GPU).
Batch: `queue_plans(project, plans)` lines them up in the project's jobqueue as "make" jobs;
jobqueue.run_queue paces them (the rest between jobs) and `queue_worker` runs each.
"""
from __future__ import annotations

import json
import re
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

KINDS = ("character", "item", "prop", "tileset", "effect", "portrait", "card", "ui", "map",
         "weather", "water", "background")
LABELS = {"character": "Character", "item": "Item", "prop": "Prop / building",
          "tileset": "Tileset", "effect": "Effect", "portrait": "Portraits", "card": "Card",
          "ui": "UI kit", "map": "Map", "weather": "Weather", "water": "Animated water",
          "background": "Background"}
FALLBACK = ("prop", "item", "character")   # nothing recognised: offer these, in this order

# --------------------------------------------------------------------------- lexicons

# Words that say which pipeline outright ("sword icon", "fire card", "stone wall tiles").
CUES = {
    "character": ("character", "characters", "npc", "npcs", "hero", "enemy", "enemies", "monster",
                  "monsters", "creature", "unit", "mob", "boss", "villain"),
    "item": ("icon", "icons", "item", "items", "inventory", "pickup", "loot", "weapon", "consumable"),
    "prop": ("prop", "props", "building", "buildings", "structure", "decoration", "furniture"),
    "tileset": ("tile", "tiles", "tileset", "tilesets", "terrain", "texture", "textures", "seamless",
                "autotile", "autotiles", "ground"),
    "effect": ("effect", "effects", "fx", "vfx", "spell", "spells", "particle", "particles",
               "animation"),
    "portrait": ("portrait", "portraits", "bust", "headshot", "mugshot", "face", "faces",
                 "expression", "expressions"),
    "card": ("card", "cards", "tcg", "ccg"),
    "ui": ("ui", "hud", "gui", "interface", "panel", "panels", "button", "buttons", "menu",
           "menus", "healthbar", "hotbar", "uikit", "frame", "9slice", "nineslice"),
    "map": ("map", "maps", "level", "levels", "battlefield", "battleground", "arena", "stage",
            "region", "overworld", "world"),
    "weather": ("weather", "atmosphere", "atmospheric", "overlay", "precipitation"),
    "water": (),                  # "animated" + a water word (scores): "animated lagoon water"
    "background": ("background", "backgrounds", "backdrop", "backdrops", "parallax", "sky", "skies",
                   "skybox", "skyline", "panorama", "vista", "scenery"),
}

# Things: phrase -> (kind, data). Multi-word phrases win over their words ("skeleton key" is a
# key, not a skeleton). data: body for characters, category for items, terrain for tiles,
# effect kind for effects; "also" = other kinds it could be (a small nudge for the list).
_PEOPLE = """knight mage wizard witch warlock sorcerer sorceress warrior archer ranger hunter
 rogue thief assassin ninja samurai monk priest priestess cleric paladin druid shaman bard
 necromancer barbarian soldier guard captain general king queen prince princess lord lady
 noble peasant villager farmer merchant shopkeeper innkeeper blacksmith smith baker miner
 sailor pirate viking gladiator spearman swordsman swordswoman pikeman crossbowman musketeer
 cowboy sheriff bandit outlaw mercenary adventurer explorer scientist doctor nurse physician
 healer alchemist chef cook maid butler child kid boy girl man woman person elder grandma
 grandpa goblin hobgoblin orc ogre troll elf dwarf gnome halfling hobbit fairy pixie angel
 demon devil imp vampire werewolf zombie ghoul ghost skeleton lich mummy golem robot android
 cyborg alien astronaut mermaid centaur minotaur cultist jester nun pilgrim scholar student
 teacher hag oracle seer dancer musician bandit lumberjack fisherman fisher shepherd squire
 templar inquisitor berserker valkyrie amazon kobold lizardman ratman wraith banshee spirit""".split()
_ANIMALS = """wolf dog puppy hound cat kitten fox bear horse pony donkey mule cow bull ox pig boar
 sheep ram goat deer stag elk moose rabbit bunny hare rat mouse squirrel badger raccoon otter
 beaver lion tiger leopard panther jaguar cheetah lynx hyena jackal coyote camel llama alpaca
 elephant rhino hippo giraffe zebra gorilla monkey ape bat chicken hen rooster duck goose swan
 owl eagle hawk falcon crow raven parrot penguin frog toad turtle tortoise lizard snake serpent
 crocodile alligator spider scorpion crab beetle slime dragon wyvern drake griffin gryphon
 unicorn pegasus basilisk hydra cerberus chimera manticore phoenix kraken shark whale dolphin
 fish octopus wolfhound mastiff husky warg direwolf panda koala kangaroo""".split()
_ITEMS = {
    "weapon": """sword longsword shortsword greatsword broadsword rapier sabre saber katana
      scimitar cutlass dagger knife dirk axe hatchet battleaxe greataxe mace hammer warhammer
      maul flail club cudgel morningstar spear lance pike halberd glaive trident javelin bow
      longbow shortbow crossbow arrow arrows quiver bolt staff wand sceptre scepter rod sling
      whip scythe sickle gun pistol rifle musket blunderbuss cannonball bomb grenade shuriken
      blade""",
    "armour": """shield buckler helmet helm armour armor breastplate chestplate cuirass
      gauntlet gauntlets glove gloves boots greaves bracers pauldron pauldrons mail chainmail
      cloak cape robe hood tunic belt crown tiara circlet""",
    "potion": "potion potions elixir tonic flask vial philter philtre",
    "food": """apple bread loaf cheese meat ham steak drumstick fish carrot mushroom pie cake
      cookie berry berries grapes pear banana egg honey soup stew wine ale beer mead milk
      sandwich sausage pumpkin""",
    "key": "key keys keycard",
    "misc": """ring amulet necklace pendant talisman charm gem gemstone jewel ruby emerald
      sapphire diamond crystal coin coins gold scroll book tome spellbook grimoire map compass
      lantern candle torch rope pickaxe shovel hoe rake fishingrod bag pouch satchel backpack
      bottle feather bone skull orb relic idol artifact artefact trinket bell horn flute lute
      harp drum dice letter envelope ticket badge medal trophy hourglass clock pelt hide
      ingot ore herb flower mushroom seed seeds""",
}
_PROPS = """house cottage hut cabin shack shop store inn tavern pub barn stable church chapel
 temple cathedral shrine tower castle fort fortress keep citadel palace manor mansion mill
 windmill watermill forge smithy workshop building home shed tent yurt pavilion lighthouse
 bridge gate gatehouse archway arch wall fence palisade statue monument obelisk pillar column
 fountain well tree oak pine palm willow birch bush shrub hedge rock boulder barrel keg crate
 box chest coffer cart wagon carriage boat ship rowboat raft table bench chair stool throne bed
 bookshelf bookcase wardrobe cupboard cabinet desk anvil cauldron pot vase urn lamppost lamp
 signpost sign post campfire bonfire brazier haystack scarecrow mailbox grave gravestone
 tombstone tomb coffin altar totem ruins ruin tower dock pier jetty stall market kiosk outhouse
 doghouse kennel birdhouse beehive trough log stump fern cactus flowerbed planter barricade
 sandbag sandbags door portal mushroom""".split()
_TERRAINS = {      # tilegen.TERRAINS keys
    "grass": "grass grassland meadow lawn turf",
    "dirt": "dirt mud earth soil path trail",
    "road": "road cobblestone cobblestones cobble cobbles paving pavement street",
    "stone": "stone flagstone flagstones floor dungeon slate tiles marble",
    "sand": "sand beach desert dune dunes",
    "snow": "snow tundra frost",
    "water": "water river lake sea ocean pond swamp lava",
    "cliff": "cliff cliffs cliffside rockface",
    "fence": "fence fences fencing",
    "wall": "wall walls brick bricks brickwork",
}
_EFFECTS = {       # effects.EFFECT_KINDS keys
    "sword_combo": "slash slashes swipe swing combo cleave",
    "hit_spark": "hit impact spark sparks clash strike",
    "big_explosion": "explosion explosions blast boom detonation kaboom",
    "fireball": "fireball fireballs",
    "fire_burst": "burst inferno",
    "fire": "flame flames blaze",
    "ice_shard": "shard shards icicle icicles",
    "frost_nova": "nova",
    "lightning_bolt": "lightning thunder thunderbolt zap",
    "holy_light": "heal healing holy blessing",
    "poison_cloud": "poison toxic venom miasma",
    "dark_magic": "curse hex",
    "buff_aura": "aura buff",
    "smoke": "smoke puff",
    "dust": "dust",
    "splash": "splash",
    "sparkle": "sparkle sparkles twinkle glitter shimmer",
    "magic_circle": "",
    "level_up_burst": "",
    "arrow_trail": "",
}
# multi-word things (checked before single words)
_PHRASES = {
    "skeleton key": ("item", {"category": "key"}), "card table": ("prop", {}),
    "magic circle": ("effect", {"effect": "magic_circle"}),
    "summoning circle": ("effect", {"effect": "magic_circle"}),
    "rune circle": ("effect", {"effect": "magic_circle"}),
    "level up": ("effect", {"effect": "level_up_burst"}),
    "level-up": ("effect", {"effect": "level_up_burst"}),
    "arrow trail": ("effect", {"effect": "arrow_trail"}),
    "chain lightning": ("effect", {"effect": "chain_lightning"}),
    "lightning bolt": ("effect", {"effect": "lightning_bolt"}),
    "frost nova": ("effect", {"effect": "frost_nova"}),
    "ice nova": ("effect", {"effect": "frost_nova"}),
    "ice shard": ("effect", {"effect": "ice_shard"}),
    "ice shards": ("effect", {"effect": "ice_shard"}),
    "fire burst": ("effect", {"effect": "fire_burst"}),
    "dark magic": ("effect", {"effect": "dark_magic"}),
    "shadow magic": ("effect", {"effect": "dark_magic"}),
    "holy light": ("effect", {"effect": "holy_light"}),
    "poison cloud": ("effect", {"effect": "poison_cloud"}),
    "crossbow bolt": ("item", {"category": "weapon"}),
    "health bar": ("ui", {}), "mana bar": ("ui", {}), "dialogue box": ("ui", {}),
    "dialog box": ("ui", {}), "text box": ("ui", {}), "inventory slot": ("ui", {}),
    "inventory slots": ("ui", {}), "ui kit": ("ui", {}), "title screen": ("ui", {}),
    "health potion": ("item", {"category": "potion"}),
    "mana potion": ("item", {"category": "potion"}),
    "rock face": ("tileset", {"terrain": "cliff"}),
    "stone wall": ("prop", {"also": ("tileset",), "terrain": "wall"}),
    "brick wall": ("prop", {"also": ("tileset",), "terrain": "wall"}),
    "castle wall": ("prop", {"also": ("tileset",), "terrain": "wall"}),
    "wooden fence": ("prop", {"also": ("tileset",), "terrain": "fence"}),
    "picket fence": ("prop", {"also": ("tileset",), "terrain": "fence"}),
    "dirt path": ("tileset", {"terrain": "dirt"}), "dirt road": ("tileset", {"terrain": "dirt"}),
    "stone floor": ("tileset", {"terrain": "stone"}),
    "dungeon floor": ("tileset", {"terrain": "stone"}),
    "wooden floor": ("tileset", {"terrain": "fence"}),
    "lamp post": ("prop", {}), "treasure chest": ("prop", {}), "magic staff": ("item", {"category": "weapon"}),
    "town hall": ("prop", {}), "guard tower": ("prop", {}), "wizard tower": ("prop", {}),
    "camp fire": ("prop", {"also": ("effect",)}), "pine tree": ("prop", {}),
    "space marine": ("character", {"body": "person"}), "sea serpent": ("character", {"body": "animal"}),
    "giant spider": ("character", {"body": "animal"}),
    # weather / atmosphere (weather.KINDS)
    "god rays": ("weather", {"weather": "god_rays"}), "god ray": ("weather", {"weather": "god_rays"}),
    "light shafts": ("weather", {"weather": "god_rays"}), "light rays": ("weather", {"weather": "god_rays"}),
    "sun rays": ("weather", {"weather": "god_rays"}), "sunbeams": ("weather", {"weather": "god_rays"}),
    "falling leaves": ("weather", {"weather": "leaves"}), "autumn leaves": ("weather", {"weather": "leaves"}),
    "drifting leaves": ("weather", {"weather": "leaves"}),
    "cherry blossoms": ("weather", {"weather": "petals"}), "cherry blossom": ("weather", {"weather": "petals"}),
    "falling petals": ("weather", {"weather": "petals"}),
    "falling snow": ("weather", {"weather": "snow", "terrain": "snow"}),
    "sand storm": ("weather", {"weather": "sandstorm"}), "dust storm": ("weather", {"weather": "sandstorm"}),
    "dust motes": ("weather", {"weather": "dust"}), "floating dust": ("weather", {"weather": "dust"}),
    "cloud shadows": ("weather", {"weather": "cloud_shadows"}),
    "lightning storm": ("weather", {"weather": "lightning"}),
    "rising embers": ("weather", {"weather": "embers"}),
    # animated water (water.THEMES)
    "ocean waves": ("water", {"theme": "ocean"}), "sea waves": ("water", {"theme": "ocean"}),
    "lava flow": ("water", {"theme": "lava", "also": ("tileset",), "terrain": "water"}),
    # backgrounds (backgrounds.SCENES)
    "mountain range": ("background", {"scene": "mountains"}),
}

_WEATHER = {       # weather.KINDS keys: single words
    "rain": "rain rainy raining rainfall drizzle downpour thunderstorm monsoon",
    "snow": "snow snowfall snowing snowflake snowflakes blizzard",
    "fog": "fog foggy mist misty haze hazy",
    "fireflies": "fireflies firefly",
    "petals": "petals sakura",
    "embers": "embers",
    "sandstorm": "sandstorm",
}
_WATER_THEMES = {  # water.THEMES keys
    "lagoon": "lagoon lagoons tropical turquoise",
    "ocean": "ocean sea seas waves wave surf",
    "river": "river stream creek brook",
    "swamp": "swamp marsh bog murky",
    "lava": "lava magma molten",
}
_SCENES = {        # backgrounds.SCENES keys
    "mountains": "mountain mountains peaks alps hills",
    "forest": "forest forests woods woodland jungle trees",
    "autumn": "autumn fall",
    "coast": "coast coastal beach seaside sea ocean island islands shore",
    "desert": "desert dunes dune canyon mesa mesas",
    "snowfield": "snowfield snowy winter tundra arctic glacier",
}
_TOD = {"night": "night nighttime midnight moonlit moonlight nocturnal",
        "dawn": "dawn sunrise morning daybreak",
        "noon": "noon midday daytime day",
        "sunset": "sunset dusk evening twilight"}
_ANIMATE = {"animated", "animating", "moving", "flowing", "rippling", "looping", "animation"}
_WATERISH = {"water", "lagoon", "ocean", "sea", "river", "lake", "pond", "waves", "lava", "swamp",
             "stream"}

_ELEMENTS = {      # word -> spell theme (fxcore.THEME_KEYS)
    "fire": "fire", "flame": "fire", "flaming": "fire", "fiery": "fire", "burning": "fire",
    "lava": "fire", "magma": "fire", "inferno": "fire", "ember": "fire", "embers": "fire",
    "ice": "ice", "icy": "ice", "frost": "ice", "frozen": "ice", "frosty": "ice", "snow": "ice",
    "cold": "ice", "blizzard": "ice",
    "lightning": "lightning", "thunder": "lightning", "electric": "lightning", "storm": "lightning",
    "shock": "lightning", "static": "lightning",
    "holy": "holy", "light": "holy", "divine": "holy", "radiant": "holy", "golden": "holy",
    "sacred": "holy", "heal": "holy", "healing": "holy",
    "poison": "poison", "toxic": "poison", "acid": "poison", "venom": "poison",
    "venomous": "poison", "plague": "poison",
    "dark": "dark", "shadow": "dark", "void": "dark", "death": "dark", "necrotic": "dark",
    "cursed": "dark", "evil": "dark", "unholy": "dark", "blood": "dark",
    "arcane": "arcane", "magic": "arcane", "mystic": "arcane", "astral": "arcane",
    "cosmic": "arcane", "mana": "arcane",
    "nature": "nature", "earth": "nature", "leaf": "nature", "vine": "nature", "forest": "nature",
    "plant": "nature", "druidic": "nature",
    "water": "water", "wave": "water", "aqua": "water", "tidal": "water", "ocean": "water",
    "steel": "steel", "iron": "steel", "metal": "steel", "wind": "steel",
}
# what each element casts when only "spell" / "magic" says it's an effect
ELEMENT_SPELL = {"fire": "fire_burst", "ice": "ice_shard", "lightning": "lightning_bolt",
                 "holy": "holy_light", "poison": "poison_cloud", "dark": "dark_magic",
                 "arcane": "dark_magic", "nature": "holy_light", "water": "splash",
                 "steel": "sword_combo"}
# spell themes -> the classic effects' themes (effects.THEMES)
CLASSIC_THEME = {"lightning": "electric", "fire": "fire", "ice": "ice", "holy": "holy",
                 "poison": "poison", "dark": "dark", "arcane": "arcane", "nature": "nature",
                 "water": "water", "steel": "steel"}
ELEMENT_FACTION = {"fire": "fire", "nature": "nature", "water": "water", "ice": "water",
                   "dark": "shadow", "poison": "shadow", "holy": "holy"}

_UI_THEMES = {     # uikit.THEMES
    "wood": "wood wooden oak timber plank planks rustic tavern medieval",
    "stone": "stone rock granite marble dungeon carved",
    "parchment": "parchment paper scroll papyrus vellum old",
    "metal": "metal metallic iron steel bronze brass gold golden silver chrome",
    "crystal": "crystal crystals glass gem gems ice icy magic arcane",
    "sci-fi": "scifi futuristic space tech cyber cyberpunk neon hologram holographic robot",
}
_MAP_TEMPLATES = {  # maplib.TEMPLATES
    "forest": "forest woods woodland jungle grove river stream creek valley",
    "harbour": "harbour harbor port dock docks coast coastal beach seaside bay island lake sea",
    "castle": "castle fortress fort keep citadel siege stronghold walls",
    "snowfield": "snow snowfield snowy winter tundra frozen ice glacier arctic",
    "village": "village town hamlet city settlement farm farmland town square",
    "flat": "flat plain plains field fields meadow arena open grassland",
}
_EXPRESSIONS = {    # word -> portraits.ALL_EXPRESSIONS
    "neutral": "neutral calm", "happy": "happy smiling smile laughing joyful cheerful glad",
    "angry": "angry mad furious rage enraged scowling", "sad": "sad crying tearful unhappy",
    "surprised": "surprised shocked astonished amazed", "determined": "determined focused",
    "smug": "smug smirking cocky", "hurt": "hurt injured pained wounded",
    "thinking": "thinking pondering thoughtful",
}
_SIZES = {"small": "small tiny little mini baby young", "large": "big large huge giant massive great"}
_ARTICLES = {"a", "an", "the", "some", "of", "with", "and", "in", "on", "for", "my", "our", "to"}


def _words(spec: dict) -> dict:
    return {w: k for k, ws in spec.items() for w in ws.split()}


_THINGS: dict[str, tuple[str, dict]] = {}
for _w in _PEOPLE:
    _THINGS[_w] = ("character", {"body": "person"})
for _w in _ANIMALS:
    _THINGS.setdefault(_w, ("character", {"body": "animal"}))
for _w, _c in _words(_ITEMS).items():
    _THINGS.setdefault(_w, ("item", {"category": _c}))
for _w in _PROPS:
    _THINGS.setdefault(_w, ("prop", {}))
for _w, _t in _words(_TERRAINS).items():
    if _w in _THINGS:          # "wall", "fence", "stone" are props or tiles
        k, d = _THINGS[_w]
        _THINGS[_w] = (k, {**d, "terrain": _t, "also": ("tileset",)})
    else:
        _THINGS[_w] = ("tileset", {"terrain": _t})
for _w, _e in _words(_EFFECTS).items():
    if _w in _THINGS:
        k, d = _THINGS[_w]
        _THINGS[_w] = (k, {**d, "effect": _e, "also": (*d.get("also", ()), "effect")})
    else:
        _THINGS[_w] = ("effect", {"effect": _e})
# deliberate choices for the shared words
_THINGS["bolt"] = ("effect", {"effect": "lightning_bolt", "category": "weapon", "also": ("item",)})
_THINGS["torch"] = ("prop", {"category": "misc", "effect": "fire", "also": ("item", "effect")})
for _w in ("campfire", "bonfire", "brazier", "candle"):
    _k, _d = _THINGS[_w]
    _THINGS[_w] = (_k, {**_d, "effect": "fire", "also": (*_d.get("also", ()), "effect")})
_THINGS["lantern"] = ("prop", {"category": "misc", "also": ("item",)})
_THINGS["crystal"] = ("item", {"category": "misc", "also": ("prop",)})
_THINGS["mushroom"] = ("item", {"category": "food", "also": ("prop",)})
_THINGS["fish"] = ("character", {"body": "animal", "category": "food", "also": ("item",)})
_THINGS["map"] = ("item", {"category": "misc"})          # only when "map" isn't the cue
_THINGS["tiles"] = ("tileset", {"terrain": "stone"})
_THINGS["rock"] = ("prop", {"terrain": "cliff", "also": ("tileset",)})
_THINGS["stone"] = ("tileset", {"terrain": "stone", "also": ("prop",)})
_THINGS["lava"] = ("tileset", {"terrain": "water", "also": ("effect",)})
_THINGS["heal"] = ("effect", {"effect": "holy_light"})
_THINGS["frame"] = ("ui", {})
for _w, _k in _words(_WEATHER).items():           # weather words (snow is also a terrain)
    _old = _THINGS.get(_w)
    _d = {"weather": _k}
    if _old is not None:
        _d.update({k: v for k, v in _old[1].items() if k != "also"})
        _d["also"] = tuple(dict.fromkeys((*_old[1].get("also", ()), _old[0])))
    _THINGS[_w] = ("weather", _d)
_THINGS["lagoon"] = ("water", {"theme": "lagoon", "terrain": "water", "also": ("tileset",)})
_THINGS["waves"] = ("water", {"theme": "ocean", "terrain": "water", "also": ("tileset",)})
for _w in ("mountains", "mountain"):
    _THINGS[_w] = ("background", {"scene": "mountains"})

_ELEMENT_OF = _ELEMENTS
_UI_THEME_OF = _words(_UI_THEMES)
_TEMPLATE_OF = _words(_MAP_TEMPLATES)
_EXPRESSION_OF = _words(_EXPRESSIONS)
_SIZE_OF = _words(_SIZES)
_WATER_THEME_OF = _words(_WATER_THEMES)
_SCENE_OF = _words(_SCENES)
_TOD_OF = _words(_TOD)
_CUE_OF = {w: k for k, ws in CUES.items() for w in ws}


def tokens(text: str) -> list[str]:
    t = (text or "").lower().replace("sci-fi", "scifi").replace("sci fi", "scifi")
    t = t.replace("9-slice", "9slice").replace("nine-slice", "nineslice")
    t = t.replace("ui-kit", "uikit").replace("health-bar", "healthbar")
    return re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)*", t)


def _singular(w: str) -> list[str]:
    out = [w]
    if w.endswith("ves") and len(w) > 4:
        out += [w[:-3] + "f", w[:-3] + "fe"]
    if w.endswith("ies") and len(w) > 4:
        out.append(w[:-3] + "y")
    if w.endswith("es") and len(w) > 3:
        out.append(w[:-2])
    if w.endswith("s") and len(w) > 3:
        out.append(w[:-1])
    if w == "men":
        out.append("man")
    if w == "women":
        out.append("woman")
    return out


def _thing(w: str):
    for s in _singular(w):
        if s in _THINGS:
            return s, _THINGS[s]
    return None, None


@dataclass
class Hit:
    pos: int           # word position (the last word of a phrase)
    words: str
    kind: str
    data: dict
    cue: bool = False


def scan(text: str) -> list[Hit]:
    """Cue words and things, in reading order (phrases first, then single words)."""
    toks = tokens(text)
    used = [False] * len(toks)
    hits: list[Hit] = []
    for phrase in sorted(_PHRASES, key=lambda p: -len(p.split())):
        pw = tokens(phrase)
        n = len(pw)
        for i in range(len(toks) - n + 1):
            if any(used[i:i + n]):
                continue
            if all(any(s == p for s in _singular(t)) for t, p in zip(toks[i:i + n], pw)):
                kind, data = _PHRASES[phrase]
                hits.append(Hit(i + n - 1, phrase, kind, dict(data), cue=kind == "ui"))
                for j in range(i, i + n):
                    used[j] = True
    for i, w in enumerate(toks):
        if used[i]:
            continue
        cue = _CUE_OF.get(w)
        if cue is None:
            for s in _singular(w):
                cue = _CUE_OF.get(s)
                if cue:
                    break
        if cue:
            hits.append(Hit(i, w, cue, {}, cue=True))
            continue
        word, found = _thing(w)
        if found:
            hits.append(Hit(i, word, found[0], dict(found[1])))
    return sorted(hits, key=lambda h: h.pos)


# --------------------------------------------------------------------------- plans

@dataclass
class Plan:
    """What to make: the pipeline (kind), its name, the words for the GPU and its settings."""
    kind: str
    name: str
    prompt: str = ""
    params: dict = field(default_factory=dict)
    text: str = ""                    # what was typed
    why: str = ""                     # why the router picked this

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Plan":
        known = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**known)

    @property
    def label(self) -> str:
        p = self.params
        if self.kind == "character":
            return f"Character ({p.get('body', 'person')})"
        if self.kind == "item":
            return f"Item ({p.get('category', 'misc')})"
        if self.kind == "prop":
            return "Building" if p.get("building") else "Prop"
        if self.kind == "tileset":
            return f"Tileset ({p.get('terrain', 'grass')})"
        if self.kind == "effect":
            return f"Effect ({p.get('effect', '')}, {p.get('theme') or 'own colours'})"
        return LABELS[self.kind]

    def gpu_minutes(self, style=None) -> float:
        return gpu_minutes(self, style)

    def summary(self, style=None) -> str:
        return summary(self, style)


@dataclass
class Guess:
    kind: str
    score: float
    plan: Plan


def _style(style=None):
    if style is not None:
        return style
    from .assets import project_style
    try:
        return project_style()
    except Exception:  # noqa: BLE001 - a broken project still routes
        from .assets import Style
        return Style()


def _clean_words(text: str, drop_kind: str | None = None) -> list[str]:
    """The words minus cue words of the chosen kind (and "a", "the"…) at the ends."""
    toks = tokens(text)
    if drop_kind:
        cues = set(CUES.get(drop_kind, ()))
        if drop_kind == "ui":
            cues |= {"kit"}
        if drop_kind == "map":
            cues |= {"template"}
        if drop_kind == "effect":
            cues |= {"spell", "spells"}
        if drop_kind == "background":        # "night sky" stays a sky in the name
            cues -= {"sky", "skies", "skyline"}
        toks = [t for t in toks if t not in cues]
    while toks and toks[0] in _ARTICLES:
        toks.pop(0)
    while toks and toks[-1] in _ARTICLES:
        toks.pop()
    return toks


def _head_text(text: str) -> str:
    """The part before ' with ', ' that ', a comma…: "tavern with a thatched roof" -> tavern."""
    return re.split(r",|;|\(|\bwith\b|\bthat\b|\bwhich\b|\bwho\b|\bholding\b|\bwearing\b|"
                    r"\bon\b|\bin\b|\bnear\b|\bfor\b|\bbehind\b|\bat\b|\bunder\b|\bnext to\b|\briding\b|"
                    r"\bcarrying\b|\bwielding\b|\bfrom\b|\bby\b", text, maxsplit=1, flags=re.I)[0]


def make_name(text: str, kind: str, limit: int = 32) -> str:
    words = _clean_words(_head_text(text), kind) or _clean_words(text, kind) or [LABELS[kind]]
    name = " ".join(words)
    if len(name) > limit:
        name = name[:limit].rsplit(" ", 1)[0]
    name = name[:1].upper() + name[1:]
    if kind == "tileset":
        name += " tiles"
    elif kind == "ui":
        name += " UI kit"
    elif kind == "map":
        name += " map"
    elif kind == "background" and not name.lower().endswith("background"):
        name += " background"
    return name


def _prompt(text: str, kind: str) -> str:
    """The words for the GPU: what was typed minus the kind's cue words."""
    toks = set(CUES.get(kind, ()))
    if kind == "effect":
        toks |= {"spell", "spells"}
    words = [w for w in re.split(r"(\s+)", text.strip())]
    out = "".join(w for w in words if w.lower().strip(",.;:!?") not in toks)
    return re.sub(r"\s+", " ", out).strip(" ,;") or text.strip()


def _first(hits, kind, key):
    for h in reversed(hits):          # the head (last) wins
        if key in h.data and (kind is None or h.kind == kind or kind in h.data.get("also", ())):
            return h.data[key]
    return None


def _element(toks: list[str]) -> str | None:
    found = [_ELEMENT_OF[t] for t in toks if t in _ELEMENT_OF]
    return found[0] if found else None


def _library_names(kind: str) -> list[tuple[str, object]]:
    """(name, thing) of what's already made, longest names first (to spot "Geela portrait")."""
    out = []
    try:
        if kind == "character":
            from . import library
            out = [(c.name, c) for c in library.list_characters()]
        else:
            from .assets import list_assets
            out = [(a.name, a) for a in list_assets(kind)]
    except Exception:  # noqa: BLE001 - no project yet: nothing made
        return []
    return sorted(out, key=lambda t: -len(t[0]))


def _in_library(text: str, kind: str):
    low = " " + " ".join(tokens(text)) + " "
    for name, thing in _library_names(kind):
        n = " ".join(tokens(name))
        if n and f" {n} " in low:
            return thing
    return None


def plan_for(kind: str, text: str, hits: list[Hit] | None = None, style=None) -> Plan:
    """The plan for making `text` with the `kind` pipeline (all settings filled in)."""
    if kind not in KINDS:
        raise ValueError(f"Unknown kind '{kind}' (use {', '.join(KINDS)})")
    hits = scan(text) if hits is None else hits
    st = _style(style)
    toks = tokens(text)
    name = make_name(text, kind)
    prompt = _prompt(text, kind)
    elem = _element(toks)
    size_word = next((_SIZE_OF[t] for t in toks if t in _SIZE_OF), None)
    p: dict = {}
    if kind == "character":
        body = _first(hits, "character", "body") or "person"
        p["body"] = body
        p["drawn"] = True            # drawn by the GPU (people and animals); 3D render without it
        p["review"] = True           # drawn: the front first, the rest after the user's OK
        p["size"] = {"small": "small", "large": "large"}.get(size_word or "", "") or \
            ("normal" if body == "person" else "medium")
        p["directions"] = int(getattr(st, "directions", 16))
        from .diorama import CLASSES
        p["role"] = next((t for t in reversed(toks) if t in CLASSES), "")
    elif kind == "item":
        p["category"] = _first(hits, "item", "category") or "misc"
        p["size"] = {"small": "small", "large": "large"}.get(size_word or "", "medium")
        p["world"] = True
        p["variations"] = 3
    elif kind == "prop":
        from .objects import guess_prop
        hp, dirs = guess_prop(text)
        p["height_people"] = float(hp)
        p["building"] = hp >= 3.0
        st_dirs = int(getattr(st, "directions", 16))
        p["directions"] = 1 if st_dirs <= 2 else int(dirs)
        p["look"] = "drawn"
        p["variations"] = 3
    elif kind == "tileset":
        terrain = _first(hits, "tileset", "terrain")
        if terrain is None:
            terrain = {"ice": "snow", "water": "water", "nature": "grass", "fire": "water"}.get(
                elem or "", "grass")
        p["terrain"] = terrain
        p["words"] = _prompt(text, "tileset")
        p["count"] = 3
    elif kind == "effect":
        fx = _first(hits, "effect", "effect")
        if fx is None:                         # "fire spell", "ice magic": the element's spell
            fx = ELEMENT_SPELL.get(elem or "", "dark_magic")
        elif fx == "fire" and any(t in ("spell", "spells", "cast", "magic") for t in toks):
            fx = "fire_burst"
        if fx in ("fire_burst", "fireball", "fire") and elem is None:
            elem = "fire"
        from .effects import SPELL_KINDS
        own = _own_theme(fx)
        theme = elem or own
        if fx not in SPELL_KINDS:
            theme = CLASSIC_THEME.get(theme, theme)
        p["effect"] = fx
        p["theme"] = theme
        p["size"] = {"small": "small", "large": "large"}.get(size_word or "", "medium") \
            if fx in SPELL_KINDS else ""
        p["paint"] = False
    elif kind == "portrait":
        exprs = []
        for t in toks:
            e = _EXPRESSION_OF.get(t)
            if e and e not in exprs:
                exprs.append(e)
        if not exprs:
            from .portraits import EXPRESSIONS
            exprs = [e for e in EXPRESSIONS if e != "neutral"]
        p["expressions"] = ", ".join(["neutral", *[e for e in exprs if e != "neutral"]])
        ch = _in_library(text, "character")
        p["character"] = getattr(ch, "name", "") if ch is not None else ""
        drop = set(CUES["portrait"]) | set(_EXPRESSION_OF)
        who = [w for w in _clean_words(_head_text(text) or text) if w not in drop]
        p["who"] = " ".join(who) or "a hero"
        p["style"] = "painted"
        name = (p["character"] or " ".join(who).title() or "Hero") + " portraits"
        prompt = p["who"]
    elif kind == "card":
        subject = _prompt(text, "card")
        p["subject"] = subject
        src = None
        for k in ("character", "item", "prop"):
            src = _in_library(text, k)
            if src is not None:
                p["source"] = k
                p["source_name"] = src.name
                break
        else:
            p["source"] = "words"
            p["source_name"] = ""
        p["faction"] = ELEMENT_FACTION.get(elem or "", "neutral")
        things = [h for h in hits if not h.cue and h.kind in ("character", "item", "prop")]
        head = things[-1].kind if things else None
        p["type"] = {"character": "Unit", "item": "Item", "prop": "Relic"}.get(head, "Spell")
        p["rarity"] = next((r for r in ("legendary", "epic", "rare", "common") if r in toks), "common")
        p["art_style"] = "pixel" if "pixel" in toks else "painted"
        p["variations"] = 3
        name = make_name(text, "card")
    elif kind == "ui":
        p["theme"] = next((_UI_THEME_OF[t] for t in toks if t in _UI_THEME_OF), "wood")
        p["scale"] = 0
        name = (p["theme"].title() if p["theme"] != "sci-fi" else "Sci-fi") + " UI kit"
    elif kind == "map":
        template = None
        for t in toks:
            if t in _TEMPLATE_OF:
                template = _TEMPLATE_OF[t]
                break
        template = template or "flat"
        from .maplib import DEFAULT_SIZES
        cols, rows = DEFAULT_SIZES.get(template, (16, 12))
        if size_word == "large":
            cols, rows = round(cols * 1.5), round(rows * 1.5)
        elif size_word == "small":
            cols, rows = max(12, round(cols * 0.75)), max(10, round(rows * 0.75))
        p.update(template=template, cols=int(cols), rows=int(rows), seed=0)
    elif kind in ("weather", "water", "background"):
        tod = next((_TOD_OF[t] for t in toks if t in _TOD_OF), "")
        if kind == "weather":
            w = _first(hits, "weather", "weather") or "rain"
            from .weather import KINDS as WK, weather_label
            if w not in WK:
                w = "rain"
            p["weather"] = w
            p["tod"] = tod
            heavy = {"heavy", "downpour", "storm", "stormy", "thunderstorm", "blizzard", "monsoon",
                     "torrential", "thick", "dense"}
            light = {"light", "gentle", "drizzle", "soft", "thin", "faint", "few"}
            p["intensity"] = 1.6 if heavy & set(toks) else (0.5 if light & set(toks) else 1.0)
            label = weather_label(w).split(" (")[0].split(" / ")[0]
            name = label + (f" ({tod})" if tod else "")
        elif kind == "water":
            theme = _first(hits, "water", "theme") or next(
                (_WATER_THEME_OF[t] for t in toks if t in _WATER_THEME_OF), "ocean")
            p["theme"] = theme
            p["tod"] = tod
            name = f"{theme.title()} water" + (f" ({tod})" if tod else "")
        else:
            scene = _first(hits, "background", "scene") or next(
                (_SCENE_OF[t] for t in toks if t in _SCENE_OF), None)
            p["scene"] = scene or "mountains"
            p["tod"] = tod or "sunset"
            # words the procedural scenes don't cover ("misty bamboo valley") -> SDXL draws it
            known = (set(CUES["background"]) | set(_SCENE_OF) | set(_TOD_OF) | _ARTICLES
                     | {"at", "time", "layers", "layer", "scene", "landscape", "view", "starry",
                        "stars", "star", "moon", "clouds", "cloudy", "sun", "distant", "far"})
            rest = [t for t in toks if t not in known]
            p["words"] = "" if (scene and not rest) or not rest else _prompt(text, "background")
    return Plan(kind, name, prompt, p, text)


def _own_theme(fx: str) -> str:
    from .effects import SPECS, SPELLS
    return SPELLS[fx][3] if fx in SPELLS else SPECS[fx][4]


# --------------------------------------------------------------------------- the router

_LLM_ROUTER = None


def set_llm_router(fn) -> None:
    """Hook for a smarter router: fn(text) -> [(kind, params or {}), ...] best first, or None
    to fall back to the rules. NOT implemented here on purpose (no network calls in this
    module): an app could back it with Claude when an Anthropic key is in keys.json. Whatever
    it returns is checked (unknown kinds dropped) and filled in by plan_for."""
    global _LLM_ROUTER
    _LLM_ROUTER = fn


def llm_router():
    return _LLM_ROUTER


def scores(text: str, hits: list[Hit] | None = None) -> dict[str, float]:
    """Rule scores per kind: a cue word 10 (later ones a hair more), the head noun (the last
    thing named) 5, other things 2, things it could also be 1, a lone element word 1."""
    hits = scan(text) if hits is None else hits
    sc = {k: 0.0 for k in KINDS}
    things = [h for h in hits if not h.cue]
    for h in hits:
        if h.cue:
            sc[h.kind] += 10 + 0.01 * h.pos
    # the head is the last thing named before "with", "in", "holding"… ("a knight in a red
    # cape" is a knight); things after that only describe it
    n_head = len(tokens(_head_text(text)))
    main = [h for h in things if h.pos < n_head] or things
    for h in things:
        if h in main:
            sc[h.kind] += 5 if h is main[-1] else 2
        else:
            sc[h.kind] += 1
        for also in h.data.get("also", ()):
            sc[also] += 1
    toks = tokens(text)
    if any(t in _ELEMENT_OF for t in toks):
        sc["effect"] += 1
    if "magic" in toks:
        sc["effect"] += 0.5
    # "a forest map with a river": the river is part of the map
    if sc["map"] >= 10:
        sc["tileset"] = min(sc["tileset"], 5)
    # "animated lagoon water", "flowing river": the animated water maker, not a still tileset
    if _ANIMATE & set(toks) and _WATERISH & {s for t in toks for s in _singular(t)}:
        sc["water"] += 10.5
    return sc


def classify(text: str, style=None, alternatives: int = 3, use_llm: bool = True) -> list[Guess]:
    """Best guess first, then up to `alternatives` other kinds it could be (each with its own
    plan). Empty text -> []."""
    if not (text or "").strip():
        return []
    hits = scan(text)
    if use_llm and _LLM_ROUTER is not None:
        try:
            picked = _LLM_ROUTER(text)
        except Exception:  # noqa: BLE001 - the rules are the fallback
            picked = None
        if picked:
            out = []
            for rank, (kind, params) in enumerate(picked):
                if kind in KINDS and all(g.kind != kind for g in out):
                    plan = plan_for(kind, text, hits, style)
                    plan.params.update({k: v for k, v in (params or {}).items() if k in plan.params})
                    plan.why = "asked the AI router"
                    out.append(Guess(kind, 100.0 - rank, plan))
            if out:
                return out[:1 + alternatives]
    sc = scores(text, hits)
    order = sorted((k for k in KINDS if sc[k] > 0), key=lambda k: (-sc[k], KINDS.index(k)))
    for k in FALLBACK:
        if len(order) >= 1 + alternatives:
            break
        if k not in order:
            order.append(k)
    out = []
    for k in order[:1 + alternatives]:
        plan = plan_for(k, text, hits, style)
        plan.why = _why(k, hits, sc[k])
        out.append(Guess(k, sc[k], plan))
    return out


def _why(kind: str, hits: list[Hit], score: float) -> str:
    cues = [h.words for h in hits if h.cue and h.kind == kind]
    things = [h.words for h in hits if not h.cue and h.kind == kind]
    if cues:
        return f"you said “{cues[-1]}”"
    if things:
        label = LABELS[kind].lower()
        return f"“{things[-1]}” is {'an' if label[0] in 'aeiou' else 'a'} {label}"
    if score > 0:
        return "it could be"
    return "a guess: nothing I know by name"


def route(text: str, style=None) -> Plan:
    """The best plan for `text` (ValueError when it's empty)."""
    guesses = classify(text, style)
    if not guesses:
        raise ValueError("Describe what to make, e.g. “a wolf” or “rusty iron sword”.")
    return guesses[0].plan


def parse_batch(text: str) -> list[str]:
    """One request per line; blank lines, "#" comments and list bullets/numbers dropped."""
    out = []
    for line in (text or "").splitlines():
        line = re.sub(r"^\s*(?:[-*•]+|\d+[.)])\s*", "", line).strip()
        if line and not line.startswith("#"):
            out.append(line)
    return out


# --------------------------------------------------------------------------- estimates

# Rough GPU minutes on Kevin's RTX 4070 8 GB, measured 2026-10-14 with ComfyUI warm and
# fp16 accumulation (generate.SPEED_FLAGS): an SDXL picture ~14 s; prompt to model (front +
# back + Hunyuan3D) ~2.5 min; drawn knight 16 directions ~23-27 drawings, ~8.5 min.
# Recalibrated from the QA run of 2026-10-15 (project "QA 2026-10-15"): tilesets and cards came
# in ~2.4x under the old estimate (swatches queue back to back, colour retries are rare), items
# 20-60% under (a chunky item's 3D model is the front picture + shape only, ~1.2 min; healing
# potion 2.0 min in all), a 3-drawing prop ~1.3 min (wooden crate).
GPU = {"sdxl": 0.25, "model3d": 2.5, "drawn16": 8.5, "rig": 2.0, "inpaint": 0.3,
       "icon": 0.2, "item3d": 1.2, "tile": 0.22, "card_art": 0.16, "prop_drawing": 0.35,
       # front-first characters: the directions after the OK'd front, per direction (QA dwarf
       # 2026-10-15: 15 directions = 25 drawings in 11.6 min) and the bending after (~1.8 min CPU)
       "drawn_dir": 0.75, "bend": 1.8}


def gpu_minutes(plan: Plan, style=None) -> float:
    p = plan.params
    k = plan.kind
    if k == "character":
        m = GPU["model3d"]
        if p.get("drawn"):           # animals ~ people (wolf 11.7 vs knight 12.5 min)
            n = max(1, int(p.get("directions", 16)))
            m += GPU["drawn16"] * (1 if p.get("review") else n) / 16   # review: the front only
        return m
    if k == "item":
        m = GPU["icon"] * max(1, int(p.get("variations", 3))) + 0.2
        if p.get("world") and p.get("category") in ("potion", "food", "armour"):
            m += GPU["item3d"]
        return m
    if k == "prop":
        if p.get("look", "drawn") == "drawn":
            if int(p.get("directions", 1)) <= 1:
                return GPU["prop_drawing"] * max(1, int(p.get("variations", 3))) + 0.25
            return GPU["sdxl"] + GPU["model3d"] + 4 * GPU["sdxl"]
        return GPU["model3d"] + GPU["sdxl"]
    if k == "tileset":
        return GPU["tile"] * max(1, int(p.get("count", 3)))
    if k == "background":
        return GPU["sdxl"] + 0.1 if p.get("words") else 0.0
    if k in ("weather", "water"):
        return 0.0
    if k == "effect":
        return 0.6 if p.get("paint") else 0.0
    if k == "portrait":
        n = len([e for e in str(p.get("expressions", "")).split(",") if e.strip()])
        m = 0.5 + GPU["inpaint"] * max(0, n - 1)
        if not p.get("character"):
            m += GPU["sdxl"] + 0.2
        return m
    if k == "card":
        return GPU["card_art"] * max(1, int(p.get("variations", 3))) + 0.12
    return 0.0


def after_ok_minutes(plan: Plan) -> float:
    """GPU minutes a drawn character still needs once its front is OK'd (review plans)."""
    p = plan.params
    if plan.kind != "character" or not (p.get("drawn") and p.get("review")):
        return 0.0
    return GPU["drawn_dir"] * max(0, int(p.get("directions", 16)) - 1)


def cpu_minutes(plan: Plan) -> float:
    if plan.kind == "character":
        return GPU["rig"] + (1.7 if plan.params.get("drawn") else 0.0)
    if plan.kind == "prop" and plan.params.get("look") == "3d":
        return 0.5
    return 0.0


def _minutes(m: float) -> str:
    if m < 0.75:
        return "under a minute"
    return f"about {max(1, round(m))} min"


def summary(plan: Plan, style=None) -> str:
    """One line: what will be made, how, and how long the GPU works on it."""
    p = plan.params
    k = plan.kind
    gpu = gpu_minutes(plan, style)
    how = ""
    if k == "character":
        how = (f"{'drawn sprites' if p.get('drawn') else '3D-rendered sprites'}, "
               f"{p.get('directions', 16)} directions, all the project's animations"
               + (", the front first for your OK" if p.get("drawn") and p.get("review") else ""))
    elif k == "item":
        how = (f"{p.get('variations', 3)} icon variations to pick"
               + (" + a world sprite" if p.get("world") else ""))
    elif k == "prop":
        dirs = int(p.get("directions", 1))
        how = (f"{p.get('look', 'drawn')}, {p.get('height_people', 1):g} people tall, "
               + (f"{p.get('variations', 3)} drawings to pick" if dirs <= 1 and p.get("look") == "drawn"
                  else f"{dirs} sides"))
    elif k == "tileset":
        how = f"{p.get('count', 3)} textures to pick, made seamless at your tile size"
    elif k == "effect":
        how = f"{p.get('size') or 'project'} size" + (", hand-painted by the GPU" if p.get("paint") else "")
    elif k == "portrait":
        who = p.get("character") or p.get("who")
        how = (f"{p.get('expressions')}, {p.get('style', 'painted')}"
               + ("" if p.get("character") else f"; a picture of {who} first (not in your library)"))
    elif k == "card":
        how = (f"{p.get('type', 'Spell')}, {p.get('faction', 'neutral')}, "
               f"{p.get('variations', 3)} art variations to pick")
    elif k == "ui":
        how = f"{p.get('theme', 'wood')} theme: panels, buttons, bars, icons, font"
    elif k == "map":
        how = f"from the {p.get('template')} template, {p.get('cols')} x {p.get('rows')}"
    elif k == "weather":
        how = (f"{p.get('weather')}, {p.get('tod') or 'neutral colours'}, a looping overlay that tiles"
               " + particles for Godot")
    elif k == "water":
        how = f"{p.get('theme')} water, looping frames + 47 shore tiles"
    elif k == "background":
        how = (f"parallax layers drawn from your words on the GPU, {p.get('tod')}" if p.get("words")
               else f"{p.get('scene')} parallax layers, {p.get('tod')}")
    cost = f"{_minutes(gpu)} of GPU" if gpu > 0 else "no GPU, a few seconds"
    later = after_ok_minutes(plan)
    if later > 0:
        cost += f" (then {_minutes(later)} more once you OK the front)"
    cpu = cpu_minutes(plan)
    if cpu > 0.75:
        cost += f" + {_minutes(cpu)[6:] if _minutes(cpu).startswith('about ') else _minutes(cpu)} on the CPU"
    return f"I'll make: {plan.label} “{plan.name}”, {how}, in your project's look — {cost}."


# Editable fields per kind: (key, label, choices tuple | "int:lo:hi" | "float:lo:hi" | bool | "text")
def fields_for(kind: str) -> list[tuple[str, str, object]]:
    from .effects import EFFECT_KINDS, THEMES
    from .spellfx import SPELL_THEMES
    from .objects import CATEGORIES, ITEM_SIZES
    from .tilegen import TERRAINS
    from .uikit import THEMES as UI_THEMES
    from .maplib import TEMPLATES
    from .cards import ART_STYLES, FACTIONS, RARITIES
    from .portraits import STYLES
    from .weather import TODS, WEATHER_KINDS
    from .water import THEMES as WATER_THEMES
    from .backgrounds import SCENES
    themes = tuple(dict.fromkeys((*SPELL_THEMES, *THEMES)))
    return {
        "character": [("body", "Body", ("person", "animal")), ("drawn", "Drawn sprites (people)", bool),
                      ("review", "Front first, wait for my OK", bool),
                      ("size", "Size", ("small", "normal", "medium", "large")), ("role", "Class", "text")],
        "item": [("category", "Category", tuple(CATEGORIES)), ("size", "Icon size", tuple(ITEM_SIZES)),
                 ("world", "World sprite too", bool), ("variations", "Variations", "int:1:6")],
        "prop": [("height_people", "Height (people)", "float:0.2:8"), ("directions", "Sides", (1, 2, 4)),
                 ("look", "Look", ("drawn", "3d")), ("variations", "Drawings", "int:1:6")],
        "tileset": [("terrain", "Terrain", tuple(TERRAINS)), ("words", "Words", "text"),
                    ("count", "Textures", "int:1:6")],
        "effect": [("effect", "Effect", tuple(EFFECT_KINDS)), ("theme", "Theme", themes),
                   ("size", "Size", ("", "small", "medium", "large", "huge")),
                   ("paint", "Hand-paint on the GPU", bool)],
        "portrait": [("who", "Who", "text"), ("character", "From library character", "text"),
                     ("expressions", "Expressions", "text"), ("style", "Shown as", tuple(STYLES))],
        "card": [("subject", "Picture", "text"), ("type", "Type", ("Unit", "Spell", "Item", "Relic")),
                 ("faction", "Faction", tuple(FACTIONS)), ("rarity", "Rarity", tuple(RARITIES)),
                 ("art_style", "Art", tuple(ART_STYLES)), ("variations", "Variations", "int:1:6")],
        "ui": [("theme", "Theme", tuple(UI_THEMES)), ("scale", "Scale (0 = project)", "int:0:4")],
        "map": [("template", "Template", tuple(TEMPLATES)), ("cols", "Columns", "int:8:64"),
                ("rows", "Rows", "int:8:64"), ("seed", "Seed", "int:0:9999")],
        "weather": [("weather", "Weather", tuple(WEATHER_KINDS)), ("tod", "Time of day", ("", *TODS)),
                    ("intensity", "Strength", "float:0.3:2")],
        "water": [("theme", "Water", tuple(WATER_THEMES)), ("tod", "Time of day", ("", *TODS))],
        "background": [("scene", "Scene", tuple(SCENES)), ("tod", "Time of day", tuple(TODS)),
                       ("words", "From words (GPU; empty = drawn by code)", "text")],
    }[kind]


# --------------------------------------------------------------------------- running

@dataclass
class Made:
    """What run_plan made. asset: the saved Asset / library Character (None until keep() for a
    tileset); variations: pictures to pick from (Paths or RGBA arrays), chosen = 1-based."""
    plan: Plan
    asset: object = None
    variations: list = field(default_factory=list)
    chosen: int = 1
    extra: dict = field(default_factory=dict)
    seconds: float = 0.0

    @property
    def name(self) -> str:
        return getattr(self.asset, "name", None) or self.plan.name

    @property
    def folder(self) -> Path | None:
        f = getattr(self.asset, "folder", None)
        return Path(f) if f else None

    @property
    def saved(self) -> bool:
        return self.asset is not None

    @property
    def waiting(self) -> bool:
        """A drawn character whose front waits for the user's OK (library.in_preview)."""
        return bool(getattr(self.asset, "waiting", False))

    @property
    def notice(self) -> str:
        """What the user must hear (e.g. the GPU failed: 3D-rendered sprites, drawings kept)."""
        return self.extra.get("notice") or getattr(self.asset, "notice", "") or ""


def _say(progress):
    return progress or (lambda text: None)


def _project(project):
    if project is not None:
        return project
    from .assets import current_project
    return current_project()


def _make_character(plan: Plan, project, say, client=None) -> Made:
    from . import library
    from .model3d import prompt_to_model
    p = plan.params
    body = p.get("body", "person")
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "model.glb"
        say(f"Making the {body} model of “{plan.prompt}” on your GPU…")
        res = prompt_to_model(plan.prompt or plan.name, out, subject=body, client=client,
                              seed=p.get("seed"), progress=lambda step, text: say(text))
        picture = getattr(res, "picture", None)
        picture = picture if picture and Path(picture).suffix.lower() == ".png" and \
            Path(picture).exists() else None
        size = p.get("size") or None
        sizes = library.SIZES.get(body, {})
        if size not in sizes:
            size = None
        drawn = bool(p.get("drawn"))
        ch = library.add_character(plan.name, getattr(res, "model", None) or out, kind=body,
                                   size=size, picture=picture, role=p.get("role", ""),
                                   progress=say, prompt=plan.prompt, drawn=drawn,
                                   preview=drawn and bool(p.get("review", False)))
    return Made(plan, ch)


def _make_item(plan: Plan, project, say, client=None) -> Made:
    from . import objects
    p = plan.params
    asset = objects.make_item(plan.name, plan.prompt, category=p.get("category", "misc"),
                              size=p.get("size", "medium"), world=bool(p.get("world", True)),
                              project=project, progress=say, client=client,
                              variations=int(p.get("variations", 3)))
    return Made(plan, asset, objects.icon_paths(asset), int(asset.extra.get("icon_choice", 1)))


def _prop_variations(asset) -> list[Path]:
    f = Path(asset.folder)
    key = lambda q: int(re.sub(r"\D", "", q.stem) or 0)            # noqa: E731
    found = sorted(f.glob("sprite_v*.png"), key=key) or sorted(f.glob("drawn_v*.png"), key=key)
    return found if len(found) > 1 else []


def _make_prop(plan: Plan, project, say, client=None) -> Made:
    from . import objects
    p = plan.params
    dirs = int(p.get("directions", 1))
    if p.get("look", "drawn") == "drawn":
        asset = objects.draw_prop(plan.name, plan.prompt, height_people=p.get("height_people"),
                                  directions=dirs, variations=int(p.get("variations", 3)),
                                  project=project, progress=say, client=client)
    else:
        asset = objects.make_prop(plan.name, plan.prompt, height_people=p.get("height_people"),
                                  directions=dirs, project=project, progress=say, client=client,
                                  look="3d")
    return Made(plan, asset, _prop_variations(asset) if dirs <= 1 else [],
                int(asset.extra.get("chosen", 1)))


def _make_tileset(plan: Plan, project, say, client=None) -> Made:
    from . import tilegen
    p = plan.params
    swatches = tilegen.make_swatches(p.get("terrain", "grass"), p.get("words", ""),
                                     count=int(p.get("count", 3)), client=client,
                                     style=project.style, say=say)
    return Made(plan, None, [sw.texture for sw in swatches], 1, {"swatches": list(swatches)})


def _make_effect(plan: Plan, project, say, client=None) -> Made:
    from .effects import make_effect
    p = plan.params
    say(f"Making the {p.get('effect')} effect…")
    asset = make_effect(p["effect"], plan.name, theme=p.get("theme") or None,
                        size=p.get("size") or None, project=project, paint=bool(p.get("paint")))
    return Made(plan, asset)


def portrait_subject(who: str, project, client=None, progress=None, seed=None) -> Path:
    """No library character to paint: a clean front picture of `who` (SDXL, the house style)
    in <project>/portrait_subjects/<slug>/ with a character.json, which portraits can paint
    from (portraits._character takes a folder). Not a library character (no sprites)."""
    from .assets import slug
    from .model3d import picture_for_model
    say = _say(progress)
    folder = Path(project.folder) / "portrait_subjects" / (slug(who) or "subject")
    folder.mkdir(parents=True, exist_ok=True)
    say(f"Drawing {who} for the portraits on your GPU…")
    img, used = picture_for_model(who, "person", client=client, seed=seed, style=project.style)
    img.save(folder / "picture.png")
    (folder / "character.json").write_text(json.dumps(
        {"name": who.title(), "kind": "person", "size": "normal", "prompt": who,
         "created": time.time(), "seed": int(used)}, indent=1), encoding="utf-8")
    return folder


def _make_portrait(plan: Plan, project, say, client=None) -> Made:
    from .portraits import make_portraits
    p = plan.params
    exprs = [e.strip() for e in str(p.get("expressions", "neutral")).split(",") if e.strip()]
    ch = None
    if p.get("character"):
        from . import library
        ch = library.get(p["character"])
    if ch is None:
        ch = portrait_subject(p.get("who") or plan.name, project, client, say)
    asset = make_portraits(ch, exprs, p.get("style", "painted"), client=client, progress=say,
                           project=project, prompt=p.get("who") or None)
    return Made(plan, asset)


def _make_card(plan: Plan, project, say, client=None) -> Made:
    from . import cards as C
    p = plan.params
    src = None
    if p.get("source") in ("character", "item", "prop") and p.get("source_name"):
        if p["source"] == "character":
            src = C.source_from("character", p["source_name"])
        else:
            from .assets import list_assets
            a = next((a for a in list_assets(p["source"], project=project)
                      if a.name == p["source_name"]), None)
            src = C.source_from(p["source"], a) if a is not None else None
    if src is None:
        src = C.source_from("words", words=p.get("subject") or plan.prompt or plan.name)
    info = C.default_info(src)
    info.name = plan.name[:24]
    info.type = p.get("type", info.type)
    info.faction = p.get("faction", "neutral")
    info.rarity = p.get("rarity", "common")
    if info.type == "Unit" and info.attack is None:
        info.attack, info.health = 2, 3
    asset = C.new_card(info, src, art_style=p.get("art_style", "painted"), project=project)
    asset = C.make_art(asset, count=int(p.get("variations", 3)), style=p.get("art_style"),
                       client=client, progress=say, project=project)
    return Made(plan, asset, C.art_variations(asset), int(asset.extra.get("art_choice", 1)))


def _make_ui(plan: Plan, project, say, client=None) -> Made:
    from . import uikit
    p = plan.params
    say(f"Drawing the {p.get('theme')} UI kit…")
    asset = uikit.new_ui(plan.name, p.get("theme", "wood"), scale=int(p.get("scale") or 0) or None,
                         project=project)
    return Made(plan, asset)


def _make_map(plan: Plan, project, say, client=None) -> Made:
    from . import maplib
    p = plan.params
    say(f"Laying out a {p.get('template')} map…")
    doc = maplib.new_map(p.get("template", "flat"), p.get("cols"), p.get("rows"), int(p.get("seed", 0)))
    asset = maplib.save_map(doc, plan.name, project=project)
    return Made(plan, asset)


def _make_weather(plan: Plan, project, say, client=None) -> Made:
    from .weather import make_weather
    p = plan.params
    say(f"Drawing the {p.get('weather')} overlay…")
    asset = make_weather(p.get("weather", "rain"), plan.name, project=project, tod=p.get("tod") or None,
                         intensity=float(p.get("intensity", 1.0)))
    return Made(plan, asset)


def _make_water(plan: Plan, project, say, client=None) -> Made:
    from .water import make_water
    p = plan.params
    say(f"Animating {p.get('theme')} water…")
    asset = make_water(plan.name, project=project, theme=p.get("theme", "ocean"), tod=p.get("tod") or None)
    return Made(plan, asset)


def _make_background(plan: Plan, project, say, client=None) -> Made:
    from .backgrounds import make_background
    p = plan.params
    words = (p.get("words") or "").strip()
    say(f"Drawing “{words}” on your GPU and splitting it into layers…" if words
        else f"Drawing the {p.get('scene')} background…")
    asset = make_background(p.get("scene", "mountains"), p.get("tod") or "sunset", plan.name,
                            project=project, prompt=words or None, client=client)
    return Made(plan, asset)


MAKERS = {"character": _make_character, "item": _make_item, "prop": _make_prop,
          "tileset": _make_tileset, "effect": _make_effect, "portrait": _make_portrait,
          "card": _make_card, "ui": _make_ui, "map": _make_map, "weather": _make_weather,
          "water": _make_water, "background": _make_background}


def run_plan(plan: Plan, project=None, progress=None, *, client=None, makers: dict | None = None) -> Made:
    """Make it with the plan's pipeline, saved in the project (tilesets wait for keep())."""
    makers = MAKERS if makers is None else {**MAKERS, **makers}
    fn = makers.get(plan.kind)
    if fn is None:
        raise ValueError(f"Nothing makes {plan.kind}s")
    proj = _project(project)
    t0 = time.monotonic()
    from .generate import gpu_retry
    with gpu_retry(say=_say(progress)):   # ComfyUI crash / driver reset: wait, carry on
        made = fn(plan, proj, _say(progress), client)
    made.seconds = time.monotonic() - t0
    made.extra.setdefault("project", proj)
    return made


def keep(made: Made, index: int | None = None, project=None) -> Made:
    """Use variation `index` (1-based; default the chosen one). Items: the icon; drawn props:
    the drawing; cards: the art; tilesets: build the set from that texture and save it."""
    index = int(index or made.chosen or 1)
    proj = project or made.extra.get("project") or _project(None)
    k = made.plan.kind
    if k == "tileset":
        from . import tilegen
        sw = made.extra["swatches"][index - 1]
        if made.asset is None or made.extra.get("kept") != index:
            gs = tilegen.build_tileset(sw.terrain, sw.texture, picture=sw.picture, seed=sw.seed,
                                       project=proj)
            made.asset = tilegen.save_tileset(gs, made.plan.name, project=proj, swatch=sw,
                                              words=made.plan.params.get("words", ""))
            made.extra["kept"] = index
            made.extra["tileset"] = gs
    elif made.variations and made.asset is not None and index != made.chosen:
        if k == "item":
            from .objects import choose_icon
            choose_icon(made.asset, index, proj)
        elif k == "prop":
            from .objects import choose_drawn
            choose_drawn(made.asset, index, proj)
        elif k == "card":
            from .cards import choose_art
            choose_art(made.asset, index, proj)
    made.chosen = index
    return made


# --------------------------------------------------------------------------- batches

def queue_plans(project, plans: list[Plan], queue=None) -> list:
    """Line the plans up in the project's creation queue (jobqueue, kind "make")."""
    from . import jobqueue
    q = queue or jobqueue.Queue(project)
    return [q.add("make", pl.name, pl.prompt, {"plan": pl.to_dict()}) for pl in plans]


def queue_worker(name: str, prompt: str, options: dict, project, makers: dict | None = None):
    """jobqueue worker for "make" jobs: run the plan, keep the first variation (nobody is
    there to pick; the generators put the best picture first: subjectcheck ranks them by
    one subject / centred / sharp and redraws sheets and scenes), return what was made
    (its .name / .folder go in the job's results)."""
    plan = Plan.from_dict(options["plan"]) if "plan" in options else route(prompt or name)
    plan.name = name or plan.name
    made = run_plan(plan, project, makers=makers)
    if not made.saved:
        keep(made, 1)
    return made


def batch_summary(jobs) -> str:
    """'7 made, 1 failed' + one line per job."""
    done = [j for j in jobs if j.status == "done"]
    failed = [j for j in jobs if j.status == "failed"]
    left = [j for j in jobs if j.status in ("pending", "running")]
    waiting = [j for j in done if any(r.get("waiting") for r in j.results)]
    head = f"{len(done)} made" + (f" ({len(waiting)} waiting for your OK)" if waiting else "") + \
        (f", {len(failed)} failed" if failed else "") + \
        (f", {len(left)} not done (stopped)" if left else "")
    lines = [head]
    for j in jobs:
        kind = j.options.get("plan", {}).get("kind", j.kind)
        mark = {"done": "✓", "failed": "✗"}.get(j.status, "…")
        where = j.results[-1].get("folder", "") if j.results else ""
        where = "/".join(Path(where).parts[-2:]) if where else ""          # items/rusty-sword
        err = f": {j.error.splitlines()[0]}" if j.error else ""
        notes = [r["notice"].splitlines()[0] for r in j.results if r.get("notice")]
        if notes and not err:
            err = f" ({notes[-1]})"
        if not err and any(r.get("waiting") for r in j.results):
            err = " (front drawn: waiting for your OK)"
        lines.append(f"{mark} {LABELS.get(kind, kind)} “{j.name}”{err}" + (f" -> {where}" if where else ""))
    return "\n".join(lines)
