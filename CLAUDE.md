# pixel-pipeline — context for Claude

Kevin "spark" Sparks's personal game-dev tool: photos and phone videos → pixel-art
sprites, animations, sheets, tilesets, `.aseprite`, Godot resources. Separate from
Pawgress. Kevin has a game-dev / pixel-art background, uses **Godot**, owns
**Aseprite**, works on Windows (RTX 4070 Laptop, 8 GB VRAM; corrected 2026-10-11, earlier notes said 4080/12 GB) and an Android phone. He
wants terse answers and fixes over polish.

## Decisions (2026-10-07)

- **Hybrid:** the free algorithmic converter is the core. AI is optional and pluggable,
  and its output always goes back through the converter (or the palette snap) so it lands on
  the exact grid and palette. PixelLab vs Retro Diffusion comparison is built (Prompt tab),
  prompt-only by Kevin's choice; he runs it with his own keys. Local models
  (Qwen-Image-Edit / Flux Kontext via ComfyUI) need fp8/GGUF on his 8 GB GPU.
- **Desktop app, Python + Qt (PySide6).** Windows first.
- **No PyTorch in the core.** PixelOE's algorithm is reimplemented in numpy/OpenCV
  (`pixelize.py`, credited in THIRD_PARTY_NOTICES).
- **Background removal = rembg's ONNX models run directly with onnxruntime**
  (`cutout.py`). No rembg dependency (it breaks PyInstaller). Masks match rembg
  bit-for-bit. Models are cached in `~/.rembg/models`.
- **`.aseprite` is written directly** from the official spec. Indexed when ≤255
  colours with binary alpha (index 0 = transparent), else RGBA. Verified against the
  `ase-parser` npm package and the round-trip reader in `tests/test_aseprite.py`. The
  web app has a JS port of the same writer.
- **Godot 4 resources** (`godot.py`): SpriteFrames and TileSet `.tres` with relative
  texture paths. Verified by loading them in Godot 4.7 headless (set `GODOT_BIN` to run
  that test).
- **Phone:** `web/index.html` is a self-contained JS port of the converter (no AI
  cut-out; colour-based Auto/Tap background removal). It is published as a claude.ai
  artifact (Goc5YHwKAowqd1bqDDCS9m) and is ALSO the UI of the Android app. `mobile/` is a
  Flutter shell that loads it and adds native pickImage / ML Kit segment / save through a
  JS bridge (`window.PixelHost` channel + `window.__pixelHostReply`). Keep that bridge
  contract stable. The file is a fragment (no doctype/html/head) because the artifact
  host wraps it; the app wraps it in `tool/sync_web.dart`. Re-publish the artifact after
  editing it.
- **Retro systems** (`systems.py`): `system` setting (nes, snes, gameboy, gbc, sms,
  genesis, gba, pico8). It applies original hardware palettes or colour depths, colours
  per sprite, sizes per subject and outline style over the subject preset; explicit
  settings still win. `look`: `readable` (default; 48x64 people, multi-palette colour
  budgets) or `authentic` (one era sprite, e.g. NES 16x24 in 3 colours). Kevin found the
  authentic sizes unrecognizable for photos, hence readable as the default. Game Boy
  (`mono`) maps photos by brightness (`tone_map`) onto 4 evenly spread DMG shades. The
  JS port in `web/index.html` must stay in sync.
- **Turnarounds** (`turnaround.py`, CLI `turnaround`, MCP `turnaround`): 4/8/16 facing
  directions. 3D models (trimesh, `3d` extra) are point-splat rendered with an orthographic
  camera (`elevation`), one scale/palette; or photos/video walking around a subject
  (`views_from_capture`: start in front, walk right; `mirror` for half captures).
  Directions follow Godot's Vector2.angle (E, ESE, SE ... clockwise); `--godot` writes an
  animation per direction plus `_facing.gd`.
- **Animated directions** (`anim3d.py`, `turnaround --animate`, MCP `turnaround(animate=)`,
  Model Studio "Animate"): own dependency-free glTF reader (skins = linear blend skinning,
  LINEAR/STEP/CUBICSPLINE TRS animations, baseColorTexture/Factor/COLOR_0 in linear space,
  uv origin top-left, alpha MASK/BLEND cut). Points are sampled once on the rest pose and
  re-posed per frame (no colour flicker); `render_points` frames all poses alike. Loop =
  start/end vertices match after removing mean offset; in-place removes the linear centroid
  drift. Later files add animations by node name (Mixamo per-animation downloads); generic
  names (mixamo.com, Take 001) take the file stem. `--colours-from` re-colours by shape
  (voxel grid) since Mixamo drops vertex colours; `model --obj` / Save all writes the .obj
  to upload. Verified on Khronos Fox/CesiumMan/RiggedSimple and in Godot 4.7 headless.
- **Auto-rig** (`autorig.py`, CLI `rig` / `turnaround --auto-rig`, MCP turnaround(auto_rig=),
  Directions tab checkbox, Prompt to Model "Animate"), 2026-10-09, toward "build PixelLab"
  locally: front silhouette grid (points -> mask, z range per cell) -> Zhang-Suen thinning ->
  crossing-number ends/junctions (8-neighbour staircases fooled plain degree counts) -> head
  / feet / hands from ends, pelvis from the crotch row (the skeleton forks too high), shoulders
  where arm paths leave the spine past the chest half-width; fallbacks = standard proportions
  (notes say which). Separate silhouette pieces are bridged. Skin = inverse distance^4 to bone
  segments, top 4. 17 joints, world-aligned rest, animations sampled 25 keys/clip; extras.loop
  marks one-shots (anim3d honours it). Verified on Khronos CesiumMan / RiggedFigure /
  BrainStem and box figures; Godot imports 17 bones + 6 animations (it doesn't loop "walk"
  for 3D use unless set in import settings; sprite exports carry loop flags).
- **Four-legged rig** (`quadruped.py`, `rig --animal`, `turnaround --auto-rig -s animal`, MCP
  turnaround(animal=), Prompt to Model Animate with Subject animal), 2026-10-11 for Kevin's dog:
  long axis = longer horizontal extent; head = end with more mass above the back line (override
  `head=`); side silhouette: belly row, legs below it split front/back at the gap, left/right by
  depth; tail = thin columns at the back. Turned head +z, left +x (same as humanoid). 19 joints
  (hips spine chest neck head tail1 tail2 + upper/lower/paw x front_l front_r back_l back_r);
  legs bind only own side/half below the body. Clips: idle walk(4-beat) run sit happy lie_down
  death, legs by 2-bone IK to planted feet. autorig's glTF writing is shared (`write_skinned`).
- **Your own photo in Prompt to Model** (2026-10-11): From a picture -> cut out -> clean-up
  window (`ui/picture_editor.py`: rub out / bring back, Ctrl+Z) -> build. `prompt_to_model(
  mirror_back=, colours=)` / CLI `model --mirror-back --colours "black, white"`: picture snapped
  to the colours (Lab, lightness stretched to the palette's first: shaded white fur stays white;
  morphological open removes glare/hairs), model vertices snapped + neighbour majority. Repaint
  = same shape, no GPU. Photos load with EXIF rotation (phone pictures are stored sideways).
  PySide teardown crashes in tests came from a lambda capturing the window in a signal and
  deleteLater without an event loop: connect methods, hand dialogs back with setParent(None).
- **Online AIs** (`online.py`, CLI `ai`, MCP `online_sprites`, desktop Prompt tab), 2026-10-09:
  Kevin wants NEW sprites from prompts and does NOT want his photos sent to any service, so
  the online providers are prompt-only. PixelLab REST v2 (Bearer; create-image-pixen /
  pixflux / bitforge sync, generate-image-v2 queued via /background-jobs/{id}; base64 may be
  raw RGBA) and Retro Diffusion v2 (X-RD-Token; POST /inferences -> poll
  /inferences/tasks/{id} -> result.base64_images; styles rd_fast/rd_plus 64-384, rd_pro
  16-256). Shapes from PixelLab's SDK, a recent v2 CLI (aviorstudio/pixellab-cli) and RD's
  api-examples repo; the cloud container's network blocks both hosts, so they were only tested
  against fakes (tests/test_online.py). Results at sprite size are snapped to the palette
  (`to_sprite`, no resampling). Keys: env or ~/.pixel-pipeline/keys.json.
- **Asset studio (2026-10-12, Kevin: "a tool to build games… characters, items, maps etc. to put
  into many games", not one game)**. Tabs: Characters, Items & props, World (Maps | Terrain
  tiles), Portraits & effects, Library, Try in a game (hd2d test bed), Sprite, More tools.
  `assets.py`: projects (~/Pixel Pipeline/Projects/<p>, PIXEL_HOME) with `Style` (palette,
  colours, outline, person_height, elevation, directions, light, tile, ppu, icon) that library
  /objects/effects/portraits/maps render with; asset store (asset.json: kind, tags,
  collections, extra) + search. `objects.py` (items: SDXL icon variations at Style.icon, world
  sprite from a 3D model of the chosen icon; props/buildings: Hunyuan model, height in people,
  directions) + `ui/objects_tab.py`. `maplib.py` (templates flat/harbour/forest/castle/
  snowfield/village, MapEdit with undo, save/load as map assets, exports godot-3d/
  godot-tilemap/godot-iso/tiled/json, load_diorama for the Game tab, install_board_extras =
  3D props) + `ui/map_editor.py`. `exports.py` presets godot-2d/godot-hd2d/unity(.meta, y
  flipped)/gamemaker(_stripN)/aseprite/generic; `reskin.py` (recolour variants);
  `jobqueue.py` (paced creation queue, variations); `ui/assets_tab.py` (Library page).
  `portraits.py` (painted busts by SDXL img2img from picture.png, expressions at 0.58 denoise:
  now face-only inpaint, see Polish wave; pixel versions) + `effects.py` (13
  procedural effects, themes) and their tabs. Character quality: `model3d.views_for_model`
  (back/side pictures by img2img over the shape rendered from that side; prompt_to_model
  views= default front+back from words) and paint_from_picture back/left/right;
  `motions.py` (cast shoot block thrust jump wave kneel victory; animal bark howl jump eat;
  add_animations; library.add_character_animations keeps old frames identical);
  `pixelclean.py` (specks, islands, nubs; on in library.render_sheets). Characters store
  their prompt. GPU verified 2026-10-12: all-four-views swordswoman (3.5 min), portraits.
- **Polish wave (2026-10-13)**:
  - Portrait expressions: face-only inpaint (YuNet landmarks -> soft face mask, crop to 1024, SDXL
    inpaint at 0.9 via generate inpaint workflow, composite so everything outside is identical,
    colour-matched); no face found (visor) -> whole-picture fallback.
  - Items: weapons/keys lie flat from the icon (thickened), chunky items stay 3D.
  - Props: hand_pixel finish (material ramps, warm lights/cool shadows), equalise_faces.
  - DRAWN props (default from words, objects.draw_prop): Kevin called the Hunyuan-rendered
    cottage awful (melty mesh posterised). Buildings and props are drawn as pixel art by SDXL +
    Pixel Art XL at the project's camera, then drop_ground (flat grey slab), converter, clean-up,
    normals.from_drawing (bevel + relief) so lights still work. SDXL draws buildings front-on
    whatever the prompt, so in angled projects a building's single view is redrawn over its
    Hunyuan shape seen from the SE corner (img2img 0.7); 4 sides = each side drawn over the
    model's view. same_size pads frames.
  - Effects: wispy smoke/dust, crown splash, arrow trail, heal ring, dark-theme rim.
  - Unity export verified in Unity 6000.3.9f1 batch mode (tools/verify_unity: 99/99 checks,
    pixels per sprite, normal map as _NormalMap, .anim clips). Fixed: metas with `sprites: []`
    on one line, sheets over 8192 px repacked into columns. GameMaker: spr_ strips, GML loader,
    not verified (not installed).
  - Updater copies with /IS /IT.
- **House style + project types (2026-10-13, Kevin: same-artist look per project, types that
  set everything up; engines Godot + Unity)**: `Style` gained style_prompt/style_negative/
  shading (soft|cel|flat -> light + hand_pixel steps)/proportions (chibi|normal|tall)/detail
  (chunky|normal|fine)/project_type/camera/tile_style/engine/export/animations (old style.json
  loads with defaults; a default Style adds NO words). `assets.style_words(style, kind)` /
  `with_style` is the one helper every generator appends: generate_sprite, icons, drawn props,
  prop/character 3D pictures + views (model3d `style=`; pixel-art terms stripped for those),
  painted portrait (neutral only; expressions repaint the face). style=None = current
  project's. `lock_palette` -> <project>/palette.hex (stored relative, read back absolute).
  `PROJECT_TYPES` hd2d/topdown-rpg/side-scroller/isometric/cards-ui/custom via
  `create_project(name, type=, size=, engine=)`; side-scroller = 2 directions (E, W:
  turnaround.direction_names(2), objects.DIRECTION_CHOICES). library.add_project_clips adds
  the type's clips motions.py knows (fall/land/climb/crouch don't exist yet: skipped).
  maplib.defaults_for(style). UI: NewProjectWizard (type -> house style page), StyleDialog
  house-style fields + palette lock, Restyle… (assets.restyle: characters rebuild, items,
  props; drawn 4-side props need the GPU).
- **Wave 3 (2026-10-13): style LoRA, cards, mocap, springs**:
  - `styletrain.py` + `ui/style_train.py` (Library -> Train style…, non-modal): trains a LoRA on
    a project's approved assets with ComfyUI's built-in TrainLoraNode (no downloads), paced, never
    touches the GPU power limit. Saved in Style.style_lora/style_trigger/style_strength.
    `generate._apply_project_style` chains it after Pixel Art XL in every built-in workflow and
    adds the trigger word (GenOptions.style_lora None = current project's, "" = off). Guide:
    docs/STYLE_TRAINING.md.
  - `cards.py` + `ui/cards_tab.py` (Portraits, cards & effects -> Cards): "card" is an asset kind
    (<project>/cards). Card art from a character/item/prop or words, frames, card scenes for
    Godot and Unity (both verified).
  - `mocap.py`: BVH parse + retarget onto the 17-joint rig; `mocap_library.json` index; files in
    $PIXEL_MOCAP or ~/Pixel Pipeline/Mocap. library.add_mocap. No mocap data downloaded yet (CMU
    plan: 47 BVH, ~33 MB, awaiting Kevin's OK).
  - `springs.py`: spring bones for capes/hair/tails (add_springs, rebake); Model Studio checkbox
    "Capes/hair/tails swing"; library.add_character_springs. Quadruped tails now hang.
  - Mocap, after Kevin OK'd the CMU download (47 BVH, 33 MB in ~/Pixel Pipeline/Mocap/cmu):
    all retarget. One-shots longer than 2.5 s are cut to their main move (mocap.action_window);
    loops need motion inside the window and weight joints by how much they move; per-clip
    `cycle` ranges in the index (dance 1.5-5 s, idles, slow walks). Characters page: "Mocap: …"
    entries in Add animation (downloaded files only). autorig._straighten_torso: capes/cloaks
    pulled the joint depth back and hunched Knight/Swordsman/Spearman/Paladin 17-31 degrees;
    the spine now stands over the knees (head may tilt 8 degrees); Skeleton.back = back-most
    surface so death lies a cape on the floor, not through it. All library people re-rigged.
  - Style training on the 4070 8 GB (2026-10-13): ComfyUI's TrainLoraNode can't fit SDXL (aborts
    under dynamic VRAM; ~10 GiB at step 0 with every lowvram/offload/bypass/dtype setting). Kevin
    OK'd kohya sd-scripts (plan A, ~120 MB): C:\AI\kohya = src/sd-scripts-main (GitHub zip) +
    pylib (pip --target --no-deps: accelerate, diffusers 0.40, bitsandbytes, opencv-headless,
    small helpers) + run_kohya.py, run by ComfyUI's embedded Python 3.13 (its torch 2.14 cu130).
    run_kohya.py must touch CUDA before the models load (else "CUDA error: out of memory" on the
    first GPU op) and needs PYTHONUTF8. styletrain.KohyaBackend (default when find_kohya(); env
    PIXEL_KOHYA): fp8 base, Pixel Art XL as --base_weights, AdamW8bit, cached latents/TE outputs,
    unet only, lr 1e-4, chunks continue via --network_weights; stops our ComfyUI first. 1.76 s/
    step at 1024; GPU ~85 C per 150-step chunk, rests bring it to ~58 C (a continuous run hit
    88 C at step 685). First style: Default project, 20 assets mirrored, 900 steps, 37 min ->
    default_style.safetensors (trigger dfltpx), before/after in Pictures/PixelPipeline.
- **Drawn characters (hybrid, 2026-10-13)** (`drawnrig.py`, `library.render_sheets/add_character(
  drawn=True)`, Characters page "Drawn sprites" checkbox): Kevin found 3D-rendered sprites
  lumpy next to SDXL-drawn ones, approved "GPU draws each direction, the skeleton bends the
  drawing". Key pose = idle frame 0, rendered on the frame grid (blocky init, k px per frame
  pixel, white bg) -> SDXL img2img 0.6, same seed, direction words -> cut out -> back to the frame
  grid exactly (no alignment guessing), one palette, IoU check/redraw. Each drawn pixel gets the
  3D point/normal/skin weights of the front surface under it (tight depth layer, bone-island
  cleanup, smoothing only between alike weights: mixing arm+cape weights shredded the drawing);
  per frame exact LBS, 4x4 samples/px, nearest-to-centre of the front surface -> no new colours;
  parts hidden at rest fill from model points coloured by the nearest direction's drawing;
  torn bits dropped; normals = skinned 3D + from_drawing relief. Keys in <char>/keys (drawn_*,
  init_*, key_*, keys.json with the camera: reused, so add_character_animations/mocap/springs
  bend the same drawings, no GPU). Knight (Neo16 test, knight-hybrid): 16 dirs ~4 min GPU +
  ~100 s bending. Limits: SDXL moving a part (sword) vs the model -> it follows the wrong bone
  in big moves; death (lying) foreshortens. Shots: Pictures/PixelPipeline/hybrid_* (_v1 = first).
  v2 (Kevin: S good, diagonals soft, keys disagree): `draw_keys_paired` (default) draws S first,
  then every view as INPAINT of the right panel of [S key | outlined blocky render] (back views
  [S | E or W | render], ~768/640 px panels); CANDIDATES (outlined init 0.6/0.72; outlined beat
  plain everywhere) scored IoU + sharpness - Lab-histogram colour distance to S - off-model
  share (`trim_outside` drops pixels >2 px off the model). `model_words` drops held things from
  the words (the init shows what the model has); `absent=("shield",)` -> negative (SDXL adds a
  shield to any knight; nothing detects that the model lacks one). Torso turned > 45 deg out of
  the drawing (death lying) -> `model_frame`: model points coloured from the drawings, majority
  filtered. Knight colour distance to S 0.33 -> 0.24 mean.
  v3 (2026-10-14, the app's default for library characters, any subject): `add_character(drawn=
  None)` = drawn when `drawnrig.available()` (ComfyUI found/answering; PIXEL_DRAWN=0/1 forces,
  conftest sets 0), else the 3D render (also on ComfyUIUnavailable). Steel came out lavender (a/b
  of neutrals +10/-8.5 vs model +4.5/+1): `neutral_tint` takes the median cast where the MODEL is
  grey off near-grey pixels (blurred weights; keeps 20% + saturated parts); fixing S fixes the
  rest (they copy its panel). Back views: order S, E, W, N, front diagonals, back diagonals;
  N painted beside [S | side], back diagonals beside [N | side] (`view_refs`), view negatives in
  the pair prompt, N's colour compared with the side (S rewarded front-like backs); BACK_
  CANDIDATES k0.6 (outlined + local contrast `crisp`, won most backs) / o0.6 / o0.72; a try that
  passes GOOD (IoU, colour, sharpness vs S, thin parts) stops the others (knight 39 -> ~27
  drawings). `thin_cover` (3x3 opening residue of the model silhouette) penalises a dropped
  staff/blade/tail. Held things: `source_pixels` (colour match within 3 px + drawn colour regions
  lying >= 35% on forearm/hand pixels take forearm/hand weights; autorig runs the knight's arm
  chain out along the blade, so hand_r is only the tip). Lying/spins: `facing_choice` bends the
  drawing whose face now looks at this camera (projected with this camera), model_frame only
  under cos 0.55; hidden-point colours skip other views' outline pixels (lying wolf was dark).
  `auto_absent`: shield/buckler unless `has_shield` (forearm+hand part widest inscribed circle
  >= 0.09 H: library paladin yes, 7 others no); animals saddle/rider/collar unless worded;
  model_words drops only absent things (a staff stays in the words). Mirror: twin direction
  mirrored when `symmetric_pair` (silhouette IoU >= 0.93 + model colours); the knight, mage and
  wolf (head turned) are not. Animals: reference E (front view of a 30 px wolf is a blob),
  SAME_WORDS, lower GOOD IoU. Resume: keys.json (camera, partial) written first, per-candidate
  files reused, keys/model.sha1 guards a cancelled add; `cancel()` -> `Cancelled`; `gpu(what)`
  context per drawing (scripts pass tools/gpu_lock); ETA = median s/drawing. `library.redraw(ch,
  ["NE"])` = new seed for those, old files to keys/old, palette kept (keys.json "palette") so the
  rest stay identical; redraw(ch, None) = all (restyle of drawn characters: assets.restyle).
  Re-skins: reskin writes "recolour" swaps into keys/keys.json, applied to the drawings' palette
  whenever they are bent again; the prompt is copied. UI: "Drawn sprites" checked when available,
  animals too, Stop button, right-click a drawn library character -> Redraw one direction /
  every direction. GPU (4070 laptop, ~22 s/drawing at 640-1024 px): knight 12.5 min first run
  (27 drawings + model load), mage 13 min, wolf 11.7 min; bending ~75-120 s. Test characters in
  Neo16 test: knight-v3, mage-v3, wolf-v3 (Default untouched); shots Pictures/PixelPipeline/
  hybrid_v3_{knight,mage,wolf}.png/.gif. Still weak: mage SE drew no staff (it crosses the robe
  in the init; thin_cover can't see that), mage back views flat (sharp 2-4 vs 5), death frames
  speckled, the 30 px wolf front views.
- **UI kit (2026-10-13)** (`uikit.py` + `ui/uikit_tab.py`, Portraits, cards & effects -> UI kit):
  asset kind "ui" (<project>/ui, in assets.KINDS). Procedural, no GPU: THEMES wood/stone/
  parchment/metal/crystal/sci-fi; Style outline/shading/detail/palette shape every piece; drawn
  at 1x, whole-number scale (default from Style.tile). 9-slice panels/window/dialogue/buttons x4/
  tabs/slots/bars/slider/scrollbar keep corners + bevels inside the margins, edges uniform, so
  engine stretching = preview. Godot ProgressBar draws fill over the whole background rect, so
  bar fills carry the frame's transparent padding. 16 px icons (optional GPU versions via
  icon_prompt + with_style), cursors, font = cards' 5x7 + the 9 missing ASCII glyphs as BMFont.
  Godot export = theme.tres (StyleBoxTexture, type variations WindowPanel/DialogueBox/Slot/
  HealthBar…, relative paths) + UIKitDemo.tscn: verified headless 4.6.1 (test). Unity export =
  sprites with spriteBorder + Editor/PixelPipelineUIKit.cs (.fnt -> Font asset, demo Canvas from
  kit.json "demo"): verified 12/12 in 6000.3.9f1 batch (tools/verify_unity/verify_uikit.py).
- **Tile variations (2026-10-13)** (`tilegen.py` + `ui/tilegen_tab.py`, World -> Tile variations):
  asset kind "tileset" (<project>/tilesets, in assets.KINDS). SDXL swatch at the project camera
  (camera_for: side-scroller -> side, isometric -> iso, else top-down), prompt via with_style
  "tile" (+ style LoRA from generate) -> seamless (CPU: 4 half-offset copies weighted sin^2/cos^2,
  zero on their own seams; inpaint_seams: offset copy's centre cross repainted) -> converter
  tileable at Style.tile, project palette. Sets: ground = maptiles 16 corner; water/cliff = 47
  blob (Godot mode 0, 8 peering bits; elliptic inner-corner notches with the sides' margins +
  edge-tapered jitter so every shared edge matches; base 0:0 has 8 map-border alternatives);
  fence/wall = 16 line tiles (mode 2, physics boxes) + base row; side/iso via tilesets.py.
  Interior variants keep a border band (shifted piece of the same picture, same colours).
  Files: .tres, .tsx wang set + example.tmj, json, texture.png, sheet.png/.json (Unity slices it;
  no clips). Godot 4.6.1 headless paints blob/line/corner sets right (test). Real-GPU run
  2026-10-13 (Pictures/PixelPipeline/tilegen_gpu_home). Kevin then asked for:
  - Fewer visible repeats: even_out (swatch picture's lowest frequencies, <=2 cycles/tile, cut;
    built-ins via maptiles.builtin_texture(even=True): the 16 px grass blotch was the diagonal
    pattern), variety() = up to 8 interiors per terrain (picture crops, every roll + the flips/
    turns `symmetries` allows by gradient skew/grain, picked by seam error, max correlation 0.3,
    a stamp or two at p 0.5) inside a 1 px (32: 2 px) band, equal probability (Godot, Tiled tsx
    <tile probability>, example maps via cell_noise). build_tileset(mix=False) = the old way.
    Metric (test): 32x32-tile fill, lattice autocorrelation 0.73-0.90 -> 0.42-0.59, blurred
    repeat contrast of the base 3.6 -> 0.7 grey levels. rebuild_tileset(asset) redoes a saved set
    on the CPU from swatch.png. Cliff full tiles now vary (base variants).
  - Maps drawn with the sets: tilegen.MapLook (newest top-down set per key: map terrain, cliff,
    fence, wall; MapDoc.tiles {key: folder, "" = built-in}, saved in map.json, editor "Tiles"
    box). Layers: ground (plain variants; shore cells get the terrain around them), corner sets
    reach into neighbouring cells (vertex = any adjacent cell), blob shores inside water cells,
    cliffs where a neighbour is 2+ levels lower, fence/wall props as line tiles (icons hidden).
    Overlay atlas = transition tile minus its plain base (darkened pixels -> black at 1-k alpha).
    Editor canvas, preview.png, godot-tilemap (ground.tres + a TileMapLayer per set on
    tilesets/<name>/overlay.tres), tiled (hidden data layer + look layers), json colour.png,
    godot-3d tops all match: Godot's composite = look.render within 1 grey level (test).
    godot-iso and the iso preview still use built-in textures.
  - Cliffs, paths, 3D terrain (2026-10-13): cliff sets = the 47 blob tiles (lit lip, grass drips,
    rock to the tile foot) + 36 face tiles (plain, no terrain; foot = 1-tile wall, mid + base = 2,
    ends W1/E2, phase = column % 3) cut from ONE strip (`CliffArt`, neo_terrain.rock_face in the
    rock texture's `neo_ramp`; seams are joints so strata run on); `face_cells` places them below
    open south sides (drop 2-3 -> 1 tile, 4+ -> 2), MapLook layer "cliff_face" (exports: node
    Cliff_Face); build_tileset(overhang=) = drips + moss. Swatch prompts: MATERIAL words lead
    (weighted) + negatives (dropped if the user's words ask for them); `colour_miss` (COLOUR_RULES:
    mean HSV + green/blue share) redraws up to `retries` (2) with a colour word, Swatch.colour_miss.
    The GPU dirt (forest floor) and water (swampy) swatches of 2026-10-13 fail it; cliff/fence pass.
    `neo_terrain.py`: neo 16-bit procedural tops/sides (diorama.textures, iso, editor built-ins)
    and `write_board_textures`: <t>_tops.png variants + _tops_n/_side_n (normals.from_drawing on
    a 3x3 copy) + <t>_lip.png overhang + terrain.json; board.gd picks a variant per cell (hash by
    weight), NEAREST + normal maps, vertex-colour wall AO, alpha-scissor lip quads. make_hd2d(tiles=
    None current project's sets | Project | MapLook | False): tops from a terrain's OWN set, walls
    in the cliff set's rock, stone tiers in the wall set's, planks from the fence set; godot-3d
    passes the map's look. Shots: Pictures/PixelPipeline/terrain_before_after.png,
    cliff_path_preview.png.
- **Game-ready 3D models (2026-10-14)** (`gamemesh.py`, `uvbake.py`, `render3d.py`, `gamemodel.py`,
  `gamemodel_export.py`, `ui/game_model_dialog.py`, Library "Export 3D model…"): heavy Hunyuan mesh
  (non-manifold, 3-4 faces per edge in places) -> narrow-band SDF on a 256 grid + skimage marching
  cubes at +0.45 voxel (closed, manifold, floaters/inner shells gone, thin capes kept) -> optional
  mirror symmetrize (rig's thigh plane; auto only above 55% mirror match) -> crease-damped Taubin ->
  batched numpy QEM (independent set of cheap edges per round, link + fold checks, weak point
  quadric (without it flat/creased parts spiked), rejected edges blocked until their area changes,
  thin sheets may go non-manifold when stuck) -> LOD0/1/2 (12k/6k/3k characters, 4k props, 2.5k
  items). UVs: charts grown in a 72 deg cone + merged, LSCM (spsolve), split when folded/overlapping/
  stretched, crumbs absorbed, raster-packed (lowest free spot, 90 deg TURN not mirror) at one texel
  density (~62% fill). Bake: colour from the heavy painted model, normals from the cleaned high-poly
  (same-side closest point), tangents exported, gutters padded. Looks smooth (1024, normal map,
  linear) / pixel (256, 4x supersampled, palette or k-means 32, NEAREST). Skin weights from the
  heavy rigged mesh (barycentric blend, top 4); rigged.glb's nodes/skins/animations are copied, so
  autorig + motions + mocap + springs clips all carry over. <asset>/model_game: lod0-2.glb, pngs,
  lod0.obj (static), <name>.ppmodel, preview.png/.gif, wire.png, info.json. Presets godot-3d-model
  (glb + .glb.import with loop flags: Godot keeps our _subresources; statics get a LOD .tscn with
  visibility ranges) and unity-3d-model (.ppmodel + Editor/PixelPipelineModelImporter.cs: a
  ScriptedImporter, no packages: LODGroup, skinned/static renderers, URP Lit or Standard, clips with
  loop flags + .controller). Verified: Godot 4.6.1 (tests/test_gamemodel.py), Unity 6000.3.9f1
  batch 42/42 (tools/verify_unity/verify_model.py: knight, archer w/ mocap+springs, dog, barrel,
  pixel potion). ~1.5-2.5 min per model on the CPU. Shots: Pictures/PixelPipeline/
  game3d_before_after.png, game3d_godot_hd2d_*.png. Test assets: project "3D test".
- **Make anything (2026-10-14, Kevin: "mostly from words, letting the GPU make it, then picking and
  tweaking")**: first tab `Make` (`ui/make_tab.py`, the app opens on it). One "Describe it" box ->
  `makeany.py` router: rule-based, no network (cue words like icon/tiles/spell/card/portrait/ui/map
  = 10, the head noun = last thing named before with/in/holding… = 5, other things 2, "could also be"
  1; small lexicons of people/animals/items by category/props/terrains/effect kinds/elements/UI
  themes/map templates/expressions; phrases first: "skeleton key", "ice nova", "stone wall").
  `classify(text)` -> Guesses (best + alternatives, each a filled `Plan`: kind, name, prompt, params);
  `summary()` = "I'll make: … — about N min of GPU" (rough GPU table from measured runs); editable
  fields per kind (`fields_for`). `run_plan` -> `Made` (asset + variations: item icons, drawn prop
  drawings, card art, tileset swatches); `keep(made, i)` picks (tilesets are saved only then); then
  `MainWindow.open_made` opens its own page. Portraits with no library character paint from a
  picture-only subject (<project>/portrait_subjects/<slug>, not a library character). Batch: one
  per line -> jobqueue kind "make" (options["plan"]), run_queue(job_ids=) paces only those, summary.
  `set_llm_router(fn)` = hook for a Claude router (not implemented, no calls). Tests:
  tests/test_makeany.py (93-phrase table, 100%), test_ui make_page_*. Shot: Pictures/PixelPipeline/
  makeany_ui.png.
- **App layout (2026-10-12, Kevin: "make it feel intuitive")**: tabs in the order you make a
  game: `1 Characters` (ModelStudio + Character Library list on the left; Kind person/animal/
  object, Size words not numbers, Name, Add to library = rig + all clips x16 + normals at the
  library scale; More settings hides the raw numbers), `2 World` (map tiles), `3 Game`
  (`ui/game_tab.py`: tick up to 6 per team from the library -> hd2d project in
  ~/Pixel Pipeline/Games/<name>, godot --import, Play launches Godot found via GODOT_BIN/PATH/
  winget), `Sprite` (From a photo | From words, basics up front, More settings folds the rest),
  `More tools` (Directions from photos, Team colours, Portrait, Simple 2D demo). Per-tab status
  hints; `MainWindow.show_tool(w)` / `current_tool()` handle the nested tabs. **Character
  Library** = `library.py`, ~/Pixel Pipeline/Characters/<slug>/ (character.json, model.glb,
  picture.png, rigged.glb, sheet.* + sheet_n.png, thumb.png); PIXEL_LIBRARY overrides (tests use
  a temp one via conftest). SIZES: person small/normal/large 80x52/96x64/112x76, animal
  64x28/72x34/88x44.
- **Desktop app** (`ui/app.py` MainWindow = QTabWidget): Sprite (the original live
  converter), Directions / Teams / Portrait / Map tiles / Godot demo (`ui/tools.py`, each a
  `ToolTab` from `ui/common.py`: slow work on a QThread via `run(job, on_done)`, `sync=True`
  in tests), Prompt to Model (`ModelStudio` embedded). Icon: the demo knight
  (`ui/icon.png`, `packaging/icon.ico`). Windows exe = PyInstaller `packaging/pixel-pipeline.spec`.
- **Godot demo** (`demo.py`, CLI `demo`, MCP `godot_demo`): writes a Godot 4 project
  (640x360 x3, integer scale, gl_compatibility): terrain map (cells set at runtime from
  map/map.json; water + half-visible edge rows blocked), units created in code (`DemoUnit`
  in unit.gd: AnimatedSprite2D + generated facing script + duplicated team material), A*
  click-to-move, spent/next turn. Default unit = `mannequin()`, a rigged knight built in
  code with `GlbWriter` (idle + walk). Verified in Godot 4.7 headless (logic) and under
  xvfb (screenshots). `make_demo` returns (files, start-screen preview).
- **Prompt to Model** (`model3d.py`, CLI `model`, MCP `prompt_to_model`, Qt
  `ui/model_studio.py` = `pixel-pipeline-model` / Ctrl+M / `--model`): prompt -> SDXL front
  picture (no pixel LoRA) -> ComfyUI native Hunyuan3D 2.1 shape (workflow mirrors ComfyUI's
  own blueprint; history key "3d") -> `paint_from_picture` (finds orientation by silhouette
  IoU over 24 rotations, projects the picture on front-visible vertices, blurred
  through-projection behind, optional back picture) -> turnaround. Untested on a real GPU;
  the fake ComfyUI in tests/test_generate.py knows the 3D nodes. ComfyUI source for node
  names: github.com/Comfy-Org/ComfyUI comfy_extras/nodes_hunyuan3d.py, nodes_save_3d.py.
- **Teams + portraits** (`teams.py`, CLI `teams` / `portrait`, MCP `team_colours` /
  `portrait`): team family = dominant saturated hue (yellows 45-110 weighted down: trim),
  recoloured in LCh keeping lightness; Godot palette-swap shader + material per team (verified
  by rendering in Godot under xvfb-run + Mesa: `xvfb-run -a godot --rendering-driver opengl3`).
  Portraits: YuNet face box (downloaded on first use), else the subject's top.
- **Map tiles** (`maptiles.py`, CLI `maptiles`, MCP `map_tiles`): built-in procedural
  seamless terrains (periodic noise) or photo/prompt textures; corner masks are smoothstep
  bilinear + shared periodic jitter so tiles agree on edges; base variants only add interior
  details (same edges). TileSet .tres format copied from Godot's own ResourceSaver output;
  alternatives 0:0/1-15 = grass next to empty cells (Godot treats empty as a terrain).
- **Side-scroller + isometric tiles** (`tilesets.py`, `maptiles --style`, MCP map_tiles(style=),
  Map tiles tab Style): side = ground texture + top texture capped on open tops (periodic cap
  depth so neighbours line up), 16 tiles by open sides (bits top 1, right 2, bottom 4, left
  8; mask m at atlas (m%4, m//4)) + 4 one-way platform pieces in row 4; Godot terrain mode 2
  (Match Sides) with side peering bits, physics_layer_0 boxes from the grass top (lip rows of
  air). Iso = W x W/2 diamonds sampled from the seamless texture (u=(a+b)/2, v=(b-a)/2),
  blocks W x W (sides x0.72 / x0.55, grass/snow sides show dirt); TileSet tile_shape 1,
  tile_layout 5 (diamond down: map (1,0) -> (W, H)), blocks in source 1 with
  texture_origin (0, H/2). Godot verified: set_cells_terrain_connect picks the right side
  tiles; xvfb render shows blocks on their cells.
- **Sprite scale research** (`docs/SPRITE_SCALE.md`, interactive page artifact
  Jv1VrzgNjgmiH4LgAQg1SR): character height as a share of screen height for 41 classic
  games by genre, and the matching sprite heights. Kevin decided all his games render at
  1080p (4K = exactly 2x), so valid pixel-art bases are 320x180 (x6), 384x216 (x5),
  480x270 (x4) and 640x360 (x3).
- **Prompt → sprite** has three routes. (1) Claude draws it: `drawing.py`
  (`sprite_from_rows`), MCP `draw_sprite`, and the web app's "Draw from a prompt" via the
  artifact `sample` capability, which is hidden in the APK. (2) Local GPU: `generate.py`
  (ComfyUI HTTP client, SDXL + Pixel Art XL LoRA; setup in `docs/LOCAL_AI.md`), CLI
  `generate`, MCP `generate_sprite`, desktop File → Generate. (3) The mix: Claude
  sketches with `draw_sprite`, then `generate_sprite(init_image=...)`. Verified on Kevin's
  laptop 2026-10-11: ComfyUI 0.39.0 portable at `C:\AI\ComfyUI_windows_portable` (start
  `run_nvidia_gpu.bat`) with SDXL base + Pixel Art XL + fp16-fix VAE (used automatically when
  installed); 2 sprites in 53 s including the first model load. No LCM LoRA yet. Kevin asked
  (2026-10-11) for no ComfyUI window: `ComfyClient.ensure_running` starts it hidden
  (CREATE_NO_WINDOW, log ~/.pixel-pipeline/comfyui.log) only at the default address, and
  atexit stops only processes we started (`start_comfyui` / `stop_started_comfyui`).
- **HD-2D tactics demo** (`hd2d.py`, `diorama.py`, CLI `hd2d`), 2026-10-11, toward Kevin's goal
  (Triangle Strategy-style games): one Godot 4.6 project, menu + three looks of one map.
  `diorama.Diorama` = height levels (1 level = half a tile), terrain per cell, props, units;
  `harbour()` = stone tiers + stairs + cobble quay + pier over water. Real 3D (`board.gd`, mode
  real): terrain meshed at runtime with SurfaceTool from map.json, units = quads with a
  Y-billboard shader drawing a sheet frame, NORMAL_MAP from `<sheet>_n.png`, direction picked
  from camera-relative facing; fov 22 telephoto, CameraAttributesPractical near+far DOF
  (tilt-shift), glow, ACES, SSAO, sunset sun + torch OmniLights, water shader with glints; Q/E
  turn 45°. Fake 3D = same board in a SubViewportContainer (stretch_shrink 3 -> 640x360), ortho
  camera snapped to whole pixels, sprites fully camera-facing. Images load with
  Image.load_from_file (no import needed); the iso look needs `godot --import` once (the editor
  does it). Units: `add_unit` writes `<name>.unit.json` (frame size, feet row, ppu 42 = a
  64 px person stands ~1.5 tiles).
- **Tactics rules** (all three looks, 2026-10-11): `diorama.CLASSES` gives hp/attack/defence/
  move/jump/reach per unit name (Unit fields override); `diorama.damage` = attack - defence//2,
  x5/4 from a higher level, min 1, no dice. Select -> purple move range (own move/jump) -> move
  or stay -> red cells = enemies within Manhattan reach -> attack (attack/hurt/death clips, else
  lunge/flash/fade), HP bars, damage numbers; done units grey, turn passes when the team is done,
  banner when one team is left. `place_units(d, blue, red)` = roster. ONE SCALE for all sheets:
  frames wide enough that height binds (people 96x64; dog 64x28, wolf 72x34, bear 88x44), else a
  spear or bow shrinks its owner. 10 GPU characters in C:/Users/spark/Pictures/PixelPipeline/chars
  (make_chars.sh / rescale.sh).
- **Sprite normal maps** (`normals.py`, `turnaround/model --normals`): camera-space normal of the
  front-most splat per pixel, OpenGL (y up), cropped/resampled with the colour sprite's own
  geometry (`_fit_box`), averaged+renormalised, never palette-snapped; `<sheet>_n.png` packed
  like the sheet. Tests share ONE QApplication (`conftest.qt_app`): a second QApplication after
  the first died with a test crashed the Windows CI run at random.
- **Lighting + glow (2026-10-13, Kevin's FX priority "torches, magic and sunsets that light up
  sprites, bloom on bright pixels")**: `glow.py` auto_glow = emissive mask (bright saturated small
  clusters + white cores; big bright patches = cloth, skipped; effect=True = all of it) ->
  `<sheet>_glow.png` (white, alpha = strength; read as alpha x max(rgb), so effects.py's own maps
  work too), written next to lit sheets (anim3d/turnaround/objects with normals), made on the fly
  by add_unit / exports when missing. HD-2D board: sprite shader light() = per-light toon bands
  (`bands`, ceil-rounded, B toggles), wrap, rim/backlight, `lift` (ambient on sprites), EMISSION from
  the glow map; a SHADOWS_ONLY twin pushed back along the light's rays (no self-shadowing);
  soft blob shadows. `hd2d.LIGHTING` dawn/noon/sunset/night -> res://lighting.json (L cycles,
  `--light=` arg, make_hd2d(lighting=), Game tab Light, CLI --light). Glow: bloom 0, HDR threshold
  1.05, screen blend, levels 2-5. Props torch/brazier/lantern/crystal (maplib.PROP_KINDS) =
  light_flicker.gd OmniLight + pixel flame shader + CPU embers; spell_light() on hits (casters
  purple/gold/green); add_effect + map.json "effects" = looping lit effects. Exports: godot-hd2d
  adds <name>_glow.png + <name>_material.tres; unity _EmissionMap secondary texture. Verified
  Godot 4.6.1 (tests/test_glow.py) + GPU shots `Pictures/PixelPipeline/hd2d_lighting_*.png`.
- **Premium spell/hit FX (2026-10-13, Kevin: "big hand-crafted-looking fire, ice, lightning and
  slash effects with many frames", rich palettes not limited)**: `spellfx.py` (14 kinds: fire_burst,
  fireball fly-loop+impact, ice_shard, frost_nova, lightning_bolt, chain_lightning, sword_combo
  hits=1-3, hit_spark angle=, holy_light (loop=True ok), poison_cloud, dark_magic, big_explosion,
  buff_aura, level_up_burst) on `fxcore.py`: shapes as heat fields at 4x on 4 materials (back,
  smoke, main, accent) -> topmost >=50% coverage, mean heat banded onto Lab-interpolated
  hue-shifted ramps (THEME_KEYS, 13-29 colours/effect), no dither; finish drops dim specks, fills
  pinholes, merges 1-step 1 px islands, outline none|dark|rim. Noise = baked fbm texture +
  cv2.remap (was 80% of the time). Timelines of phases with ms per frame (multiples of 10),
  physics on real time; fade by shrinking/eroding, not darkening (darkening went muddy brown).
  effects.render_effect/make_effect dispatch SPELL_KINDS (EFFECT_KINDS = premium + CLASSIC_KINDS);
  Effect has durations/glow/blend/tags. Files: sheet_glow.png (rgb x emission), json per-frame
  durations + meta loops/blend, Godot per-frame durations (godot.write_sprite_frames
  durations_ms), sheet_add.tres (blend_mode 1) + sheet_demo.tscn/.gd; exports read meta loops
  (Source.is_loop). Optional `fxpaint.paint_effect` (SDXL img2img over a grid of all frames, one
  call, snapped back to the effect's palette inside its own alpha; tested with the fake ComfyUI
  only). Effects tab: size small/medium/large/huge, real timing, dark/light ground, bloom preview.
  `tools/fx_preview.py` -> Pictures/PixelPipeline/fx_preview.png/.gif. Godot 4.6.1 loads it (test).
- **Weather, water, skies (2026-10-14)** (World -> Weather && sky, `ui/weather_tab.py`; kinds "weather",
  "water", "background" in assets.KINDS; exports.KIND_EXPORTS -> `atmos_export.py`). CPU, fxcore heat fields:
  `weather.py` 12 kinds (rain streaks/splashes/ripples, snow, fog banks, god_rays, leaves, petals, embers,
  fireflies, dust, sandstorm, lightning one-shot + flash list, cloud_shadows): every overlay LOOPS and TILES
  (things move whole tiles / whole sway cycles per loop, drawn again across edges), banded alpha steps, glow
  maps, TOD_TINT dawn/noon/sunset/night (None = neutral). Files = effects.write_effect sheet + particles/*.png
  strips + weather.json (Emitter list, px and y-up units) + `weather_godot.py`: <sheet>_overlay.tscn
  (CanvasLayer, weather_overlay(_add).gdshader tiles the loop at whole-pixel scale, follows the camera),
  _particles2d.tscn (GPUParticles2D per depth layer, 480x270 view), _particles3d.tscn (GPUParticles3D
  billboards; cloud shadows = drifting 2x2 Decal; lightning = Sprite3D bolt + DirectionalLight flash),
  weather_fx.gd. `water.py`: seamless looping water (integer-frequency waves + periodic warp, crest lines,
  2-frame glints), 47 shore tiles on tilegen.blob_region (tide breathes the margins, foam rides the
  waterline), side surface/body, iso diamonds; animated Godot TileSet (tile (c,r) at atlas (c*n, r), frames to
  the right; verified: 48 tiles x n frames, terrain painting) + Tiled tsx; make_water(tileset=None) matches the
  newest tilegen water set (colours, tile, shore ground). `backgrounds.py`: 6 scenes x 4 TODs, 5 layers sky/
  clouds/far/mid/near (banded sky + 1-row checker blend, sun/moon halos, stars, haze by distance, sizes scale
  with height), seamless x; background_from_prompt = SDXL (with_style "tile") -> parallax.build_parallax,
  depth model only if already on disk else guess_depth (no download). Godot Parallax2D (clouds autoscroll),
  Unity layers + .meta + PixelParallax.cs (not run in Unity). HD-2D: make_hd2d(weather="rain"|"storm"|...|
  list|asset, water="classic"|"pixel") -> `weather_hd2d.install`: res://weather/<kind>/ + board_weather.gd
  (board.gd `_build_weather`, fits emitters to the board, splashes from cell-top emission points, wet = darker
  glossy StandardMaterial3D, snow frosting) + lighting.json presets tweaked per weather; needs godot --import.
  WATER_SHADER_PIXEL: pixel-stepped ripples, depth-buffer shore foam, screen-texture reflection. Verified:
  tests/test_weather.py (Godot 4.6.1 headless loads all 36 weather scenes, TileSet, parallax, board), real GPU
  shots Pictures/PixelPipeline/hd2d_weather_*.png; previews weather_preview.png/.gif, water_preview.gif,
  parallax_preview.png (`tools/weather_preview.py`).
- **Isometric tactics look** (`iso_scene.py`, CLI `iso-demo`), 2026-10-11, one of the three looks
  of the tactics demo over `diorama.py`: `write_iso_scene(d, project, {name: sheet.png})` writes
  `iso/` (+ map/map.json if absent, project.godot only if absent). One pre-rendered column per
  cell (top + sides from its height down to level -2; tile W, level = W/4 px; side texture 1:1,
  line per level), painter's order by z_index = (col+row)*8 + layer. GDScript range = path_cost
  (test compares them in Godot). Grid +col faces SE, +row SW. Verified on Godot 4.6.1 with real
  GPU screenshots (`Pictures/PixelPipeline/hd2d_iso_test/shot_*.png`, `shot.gd` makes them).
- **GPU speed (2026-10-14, Kevin: "quality… just seeing if we can make things faster")**: rule =
  no visible loss as a default. Measured (4070 laptop): SDXL picture ~15 s of which 25-step
  sampling ~11 s, ComfyUI overhead ~3-5 s (LoRA patching under dynamic VRAM 1-2 s, a LoRA-stack
  switch ~6 s, VAE enc+dec 1.4 s at 1584x792); cold ComfyUI start ~80 s (~20 s warm) = the
  "1.3 min model load" of a drawn character, not a reload; after Hunyuan3D, SDXL reloads from disk
  (~12 s; 16 GB RAM can't hold both). `--disable-dynamic-vram` reloads every prompt, `--highvram`
  thrashes (28 s/it): keep the defaults. Defaults now: ComfyUI we start gets `--fast
  fp16_accumulation` (generate.SPEED_FLAGS, -10-15% SDXL + Hunyuan; same-seed pictures
  indistinguishable, same shape from the same picture; PIXEL_COMFY_FAST=0 off; not for training
  flags); `warm_up()` starts ComfyUI when the app opens (PIXEL_WARM=0 off); count>1 queues all
  prompts at once; picture cache ~/.pixel-pipeline/gen_cache (sha256 of the workflow minus
  filename_prefix + model file size/date; uploads are content-named; only the default local
  ComfyUI, PIXEL_GEN_CACHE=0/1, 3 GB). Opt-in only: GenOptions.draft / PIXEL_DRAFT / File ->
  Draft pictures = SDXL-Lightning 8-step LoRA (euler, sgm_uniform, cfg 1.5; ~2.5x per picture,
  knight 16 dirs 6.4 vs 12 min) - the A/B showed sprite sheets/noise in items/props/people, muddy
  drawn keys, harsh portrait expressions; tiles were competitive but missed snow/water colours.
  Rejected: 16 directions bent from 8 drawn keys (speckle, jumbled capes), candidate score
  thresholds (later tries won 4/10 times on the knight). Sheets: Pictures/PixelPipeline/speed_ab_*.
- **CPU speed + less waste (2026-10-14, same ask)**: outputs bit-identical to before (sheets,
  _n, json, aseprite, rigs, springs: sha1-compared on Neo16 knight-v3 copies; tests/test_speed.py
  pins every fast path to its plain version). Exact rewrites: pixelclean islands/spurs/
  pixel-perfect (visit only candidates, same raster order), selout (one shade per colour),
  downscale median (sort uint8), `_splat` (pose columns + buffers, box-sized post-process),
  drawn bend (float64 weights once per drawing, KD tree once), uvbake pack (one summed-area
  table per chart), LOD bakes on threads. `parallel.py`: spawned worker pool (BELOW_NORMAL,
  1 BLAS/cv2 thread each, count = min(cpus*3/8+1, 8, (free RAM-1.5 GB)/per-worker), PIXEL_WORKERS=1
  off; falls back in-process; gui/cli_main call freeze_support). Drawn bending = one task per
  direction (arrays memory-mapped); 3D sheets = staged A pose stats / B splats / C convert / D
  finish (main process 4.4 GB -> 0.55 GB; in one process only when RAM is short). `cache.py`
  (~/.pixel-pipeline/cache, PIXEL_CACHE=off|dir, PIXEL_CACHE_MB 2048, LRU): keys = input bytes +
  settings + source of the computing code: drawn drawing->frame (skips the cut-out model),
  bent rows per (direction, clip), 3D converter per view, spring sims per clip, auto-rig by
  model bytes; model_game info.json cache_key = "up to date" skip. gamemesh: fast-simplification
  (MIT, `3d` extra, spec hidden import) agg 3, LODs chained, numpy QEM fallback; A/B (4 models):
  closer (mean/p99 distance, normals, silhouette) at every LOD, bake colour error equal +-0.1,
  more slivers (0.1-2.6%) - Pictures/PixelPipeline/decimate_ab.png. Knight: drawn render 121 ->
  55 s (cold) / 14 s (warm), Add animation 133 -> 21 s; 3D render 329 -> 105 s, Add animation
  390 -> 70 s; game model 84 -> 40 s, again 0 s.
- **QA pass 2026-10-15** (project "QA 2026-10-15", Pictures/PixelPipeline/qa_all.png, qa_hd2d.png): fixed
  parallel workers never lowering priority (SetPriorityClass got the pseudo-handle as a 32-bit int; now typed
  WinDLL, test), quadruped rig on three-quarter-picture animals (`body_axis_yaw` turns a diagonal body
  straight first; the Make-tab wolf was skewed in every sprite), godot-hd2d export of a tileset (KeyError 0
  -> clear ValueError), test_updater wrote the real update.log. Open (sheets: fixed, see "One subject per picture"); router has no weather/background kinds ("rain at night" -> a drawn prop); a
  ComfyUI/driver reset mid-draw drops the drawings and silently falls back to the 3D render.
- **One subject per picture (2026-10-15, the QA's biggest quality bug)** (`subjectcheck.py`): for one-thing
  requests SDXL drew sprite sheets / several things / room dioramas in ~half the pictures. A/Bs at fixed seeds
  (7 cases x 3 seeds, scratchpad ss_ab/, sheets Pictures/PixelPipeline/single_subject_ab_prompts*.png): the
  SEED decides the layout - prompt order, weights, "one/single/isolated", dropping "game asset"/"inventory icon"
  /"concept art", short view words, CFG 4.5, Pixel Art XL 0.8, style LoRA 0.5/off, trigger position, sheet
  negatives all stayed 6-12/21; img2img from ONE soft blob in the middle (`single_init`, shape by words
  `shape_for`: weapon/bottle/building/tall/flat/person/animal, filled with the request's colour/material
  `init_colour` - a grey blob made a red apple white and a wooden shield steel) gave 21/21. Denoise 0.93
  pixel kinds (0.8 = grey ghosts), 0.97 painted 3D/portrait pictures (0.9 kept the blob as a grey silhouette;
  noise in the blob washed them out): `SINGLE_OPTIONS`. `assess(picture)` (CPU ~90 ms, no model):
  background = border colour + smooth-gradient flood seeded from it; pieces after a small close; pieces within
  3% of the main piece are its own; extras / sheet (>=5 extra pieces or 3+ similar) / scene (border not plain
  or fill > 70%) / cut off / tiny / flat grey silhouette (start-blob pictures only). `generate_single` = draw
  count, redraw failures with new seeds (budget max(2, count)), return best first (score: one subject,
  centred, sharp) -> picture_v1 / drawn_v1 / the batch's blind keep(1) is the best; asset.extra["checks"].
  Used by make_item icons, draw_prop drawings, _build_model prop picture, model3d.picture_for_model (+ the
  rembg subject_pieces check; portrait subjects + characters go through it). PIXEL_SINGLE=0 = old behaviour.
  Prompts: subject first, "concept art" -> "illustration", objects.SHEET_NEGATIVE / drawn_prompt(); style
  words for one-subject kinds drop "... effects" parts and plural "sprites" (assets.SINGLE_KINDS);
  _clean_terms drops a part whose pixel words aren't a leading prefix ("modern 16-bit pixel art" was
  "modern", "SNES-era sprite craft" was "era craft"). Measured on 28 prompts (8 items, 8 props, 4 buildings,
  3 creatures, 5 people/portrait subjects), QA project + style LoRA, seed 9100 (scratchpad ss_measure.py,
  Pictures/PixelPipeline/single_subject_ab.png): first pick one subject 14/28 detector / ~12/28 by eye ->
  28/28 (69 pictures drawn for 60 variations + 8 model pictures; 1 redraw). Still wrong by eye: leather
  boots -> a chest piece/hood, anvil -> a stone block, one odd street lamp; painted 3D pictures look flatter.
  Style LoRA + neo16 look still show. Limit: the detector can't tell a hut from a barrel (semantics).
- **Robustness fixes after the QA pass (2026-10-15)**: `generate.GpuRetry` / `with gpu_retry(say=)` (one failure
  count per job; generate_images, model3d.shape_from_picture, drawn characters, run_plan use it): ComfyUI stops
  answering or a picture fails with a CUDA/driver/OOM error (`gpu_fault`) -> `recover_comfyui` (our ComfyUI is
  restarted, also when alive after a CUDA error; backoff 2..30 s up to PIXEL_GPU_RECOVER_S=240) -> the same
  picture again (pictures of a count>1 call already made are kept). After PIXEL_GPU_RETRIES=3 failures:
  `ComfyUIGaveUp` (a ComfyUIUnavailable) with the events + log tail; conftest sets retries 0. Drawn characters:
  keys/ are KEPT on a give-up, 3D render as fallback with `Character.notice` (message box on the Characters
  and Make pages), keys.json stays `partial` -> `library.is_drawn` False, `drawing_unfinished` True,
  `finish_drawing(ch)` / right-click "Finish the drawn sprites" draws only the missing ones. Batches
  (jobqueue.run_queue): a give-up puts the job back to pending and stops the run. Logs: `rotate_log` on every
  start keeps comfyui.1.log (the crashed run) .. comfyui.5.log. Make router: kinds weather / water /
  background (makeany._WEATHER/_WATER_THEMES/_SCENES/_TOD; "animated"/"flowing" + a water word -> water;
  background words beyond scene/time of day -> SDXL background_from_prompt), opened on the Weather && sky page
  (`WeatherTab.open_asset`); animals from Make are drawn (plan drawn=True; 3D without ComfyUI). Tiles:
  road prompt = rounded irregular cobbles, bricks in the negative; `tilegen.pattern_stats/pattern_miss`
  (4-fold edge-orientation coherence for bricks on road, x horizontal/vertical balance + row/column line
  energy for grids in water/grass/dirt/sand/snow/road; QA brick road 0.45, QA lagoon 0.38/0.32 vs 0.04-0.18 for
  good ones; real GPU cobbles from the new prompt 0.21-0.30 -> BRICK_MAX 0.36, GRID_MAX 0.31), checked as
  drawn and again after seamless, redrawn with PATTERN_WORDS (Swatch.pattern_miss). GPU-verified 2026-10-15
  (after the driver update): killed our ComfyUI mid-batch -> restarted, 3/3 pictures, 86 s, crash log kept
  as comfyui.1.log; road now draws rounded cobbles; QA barrel -> 3D shape + game model in 120 s
  (Pictures/PixelPipeline/robust_tiles.png). Lagoon swatches no longer grid but come out nearly flat blue.
  Tiles v2 (same day): the whole 1024 swatch squeezed into a 32 px tile turned cobbles into grey noise, so
  `tile_crop` takes the piece whose features (`feature_size` = 2x first zero of the radial autocorrelation)
  land at FEATURE_PX in the tile (x sqrt(tile/32) for small tiles; the most terrain-coloured of 9 placements)
  before seamless; the CONVERTED tile is judged (`tile_stats`: region size in 4 lightness bands, noise 1.9 /
  squeezed cobbles 2.6 / readable 4.5+; L std; `tile_miss`), flat tiles get `lift_detail`, and water that stays
  flat or full of land (colour rule now blue >= 0.7, warm <= 0.08) becomes `neo_water_tile` (water.py ripples +
  glints in the swatch's own blue). `pattern_stats.rows` (2-fold edge coherence) catches brick courses / streaks
  (ROWS_MAX); after seamless only grids/lines are checked (the half-offset blend fakes directions on small
  crops). GPU run, QA house style (Pictures/PixelPipeline/robust_tiles_v2.png = raw | tile x3 | sample map):
  road readable stones, lagoon = neo ripples (SDXL drew islands + a palm 3/3 tries), grass fine; still weak:
  dirt tile repeats in diagonal streaks, sand faint streaks (flagged, kept), SDXL "snow" = drifts on dark
  ground, cliff drawn as courses (not checked). The house style (style LoRA + "16-bit") seems to push swatches
  toward bricks/courses.
- **Tiles v3 (2026-10-15/16, Kevin: "finished tiles are much worse than the swatches")** (`tilepaint.py`,
  `tilegen.METHODS` / `tile_method`, make_swatches(method=None)): top-down + iso default = **paint** for road/cliff
  (irregular mixed-size cobble/boulder layout at the tile size -> 16x, softened, 2x2 1024 canvas -> SDXL img2img
  denoise 0.7, style LoRA at TILE_STYLE_STRENGTH 0.35 -> `fold` (sin^2 over the canvas: wraps exactly, no ghosts)
  -> `stone_tile` (watershed stones on evened lightness, 3x3 wrap, ids by marker position; per-stone tone rank +
  3 shared tints, lit/shadow rims, glint, rounded corners, 1 px joints) + `paint_variants` (one more pass: 4 other
  layouts) = 2 GPU pictures per swatch; misses -> lower denoise -> neo layout) and **drawn** for grass/dirt/sand/
  snow/stone/water/fence/wall (code-drawn shapes at tile scale, evenly scattered details, ramp = hand ramp mixed with
  SDXL swatch colours (`picture_ramp`, SWATCH_WEIGHT 0.6, snow 0.3; only colour_miss is checked); water = neo ripples
  in the swatch blue; `drawn_variants` = other seeds). Side-scroller camera keeps **swatch** (the old way; also
  method="swatch"). Swatch.method/sources/variants_raw; build_tileset(sources=) feeds variety() (base grass gets its
  drawn variants too); save writes extra.method + swatch_variants.png; rebuild_tileset redoes paint/drawn on the CPU
  identically; builtin_swatch (no GPU, even=True) = drawn / neo. GPU A/B findings (QA project): SDXL keeps the
  init's pixel steps at 0.4-0.55 (rows of square stones), redraws round stones at 0.7; it flattens soft ground into
  blobs at any denoise and at 2x2 or 4x4 canvases (so soft ground is drawn); the LoRA at 0.35 vs 0.8 made no
  bricks in paint mode (bricks came from the free swatch prompt); the old converter's contrast downscale point-
  samples (aliasing) and prepare(tileable) re-blends an already seamless picture (ghosts) - finish_tile uses area
  downscale (tested isotropic). Sheets: Pictures/PixelPipeline/tiles_quality_ab.png (old tile | new tile | sample
  map 1x | transition 2x | HD-2D board before/after), tiles_quality_hd2d_before/after.png; v3 sets saved in
  "QA 2026-10-15" (v3-*-tiles). Weak: on the HD-2D board at noon snow reads near-plain white and sand pale (fine
  detail lost to bright light + depth of field; 2D is fine); road stones sometimes merge into odd shapes; cliff
  faces are still neo rock_face (unchanged). Tests: tests/test_tilepaint.py.
  Drawn picture-only props/items: `gamemodel.needs_shape` -> Export builds model.glb from the chosen drawing
  (prompt_to_model picture=, views front; GPU ~1.5 min, said in the dialog). makeany.GPU recalibrated from the
  QA timings (tile 0.22/texture, card art 0.16, icon 0.2 + item 3D 1.2, prop drawing 0.35). Tests:
  tests/test_robust.py (fake ComfyUI that drops connections mid-run).
- **Tiles round 2 (2026-10-16, Kevin: "work on those")**: (1) HD-2D board ground = `hd2d.TERRAIN_SHADER`
  (shaders/terrain.gdshader, ShaderMaterial per top/side, `terrain_mats`): per-texel normals (nearest) + each
  light's diffuse in `terrain_bands` (5, lighting.json; B toggles with the sprites' bands) steps, lower bands
  toward a cool shade_tint, no specular (weather wet adds some); `neo_terrain.board_albedo`: bright tops capped
  (ALBEDO_CAP snow 83 / sand 76 Lab L at p98), their steps spread (BOARD_CONTRAST 1.6 / 1.75), shadows hue-shifted
  (SHADE_SHIFT); RELIEF snow 1.5 / sand 1.3 + terrain.json "relief" (normal depth 1.6 / 1.4); noon sun 1.75 -> 1.6,
  exposure 0.93, "white" 1.6 (env.tonemap_white); tilt-shift band wider (near cam_dist-7, far +5, amount 0.12).
  Noon 1:1 crop: snow L 97 -> 87, std 11 -> 14 (drifts read), sand reads as sand. weather_hd2d tints the shader
  mats too. (2) `stone_tile` -> `split_stones`: stones bigger than STONE_AREA max (road 0.13 x tile area), aspect
  > 2.6 or solidity < 0.78 (`stone_shape`, wrap-aware) split by 2-means along a new 1 px joint (halves >= 30%);
  on the real 2026-10-15 GPU paintings odd stones (>= 12 px) road 20% -> 12%, cliff 29% -> 16% (stricter 0.8/2.3
  metric). (3) Cliff faces: `tilepaint.boulder_face` (same signature as rock_face; `face_layout` = strata bands +
  leaning wobbly cuts, big ones split in rows; stones drawn like stone_tile: tone rank, 3 tints from the cliff
  tile (`stone_tints`), dome shading, 2-pass rounding, glint, deep joints, moss) used by `cliff_art` (CliffArt
  layout/rules unchanged) and the board's natural side walls (bricks still rock_face). FACE_STYLE "strata" = old.
  (4) `tilewords.py`: words shape drawn ground: `details(terrain, words)` -> Details (moss, flowers, cracks,
  puddles, pebbles/big, ripples/dunes, fresh, footprints, tall, leaves, tufts; USES per terrain; whole words,
  phrases first); grass/dirt/sand/snow/_flag idx painters take d (plain = byte-identical to before);
  `stamp_details` = coloured stamps (moss clumps, flowers, leaves, puddles, grey pebbles). Swatch.words,
  drawn_tile/drawn_variants/builtin_swatch(words=), save_tileset stores them, rebuild_tileset redoes them; the UI
  Quick button passes the Words box. SDXL-suggested stamp shapes: not done. (5) makeany: `tile_pictures` ->
  painted terrains (road/cliff, top-down/iso) = 2 x count pictures x GPU["paint_picture"] 0.35 min (QA road
  41 s/swatch); progress_plan has a "Painting N tiles (2N pictures)" stage. Tests tests/test_tiles_round2.py.
  Sheets: Pictures/PixelPipeline/tiles_round2_ab.png (2D before/after per terrain, road/cliff stone_tile on the
  real paintings, word variants on today's SDXL swatches), tiles_round2_hd2d.png (4 times of day + noon crops).
  Note: this run SDXL took ~55-75 s per 1024 picture (ComfyUI cold start in the lock), vs 13 s before.
  Still weak: the board's road uses the QA project's saved (pre-split) road set; noon grass is richer/greener
  than before (no specular wash) - check it's wanted; puddles/footprints repeat per tile (2 per tile, variants
  help); moss on dirt reads as small bars; face strata seams are straight joints every tile row (by design for
  the cut rules).
- **Front first, then OK (2026-10-15, Kevin: "show one side then ask for approval to make all 16
  directions, or redraw if it's not up to par")**: `library.add_character(drawn=True, preview=True)` =
  picture -> model -> rig -> ONLY the reference drawing (S; animals E) + its idle bent from it
  (`drawnrig.drawn_preview`: same `_drawn_setup` cameras/framing as drawn_animation, `draw_keys_paired(
  only=(ref,))`, CPU idle via bend_direction) -> a DRAFT: keys/keys.json {"partial", "preview": {direction,
  current, attempts, frame_ms}} (survives restarts), no sheet, thumb from the key. `in_preview` /
  `Character.waiting` / `is_draft_folder`; `approve_preview` = finish_drawing (resume path: the S
  candidate files are reused byte for byte, the rest paired to them); `redraw_preview(seed=, words=,
  new_model=)` (new seed S only; words keep the seed; new_model = prompt_to_model again + re-rig, old
  tries dropped); `pick_preview(ch, n)` (keys/attempts/<n>/: drawn_S_*, key.png, idle_*.png; no GPU);
  `rest_estimate` = makeany.GPU drawn_dir 0.75 min x (n-1) + bend 1.8 min. Drafts: badge + "waiting for OK" in the Character
  Library, right-click Review / Finish; Game tab lists only finished ones; exports raise / skip
  ("skipped" in index.json); restyle skips; add animation/mocap/springs refuse. UI `ui/character_review.py`
  (CharacterReview: key big + idle playing + picture small + estimate, Looks good / Redraw this side /
  Change words… (words, house style words, "New model too") / Not now (keep draft) / Throw it away,
  earlier tries strip, Stop) in ModelStudio (stack page; "Front first, wait for my OK" in More settings,
  default on) and MakeTab (beside the plan; plan param `review`, default on; batch "Stop for my OK on each
  character" -> drafts in a "Waiting for your OK" list; jobqueue result "waiting"; batch_summary counts
  them). jobqueue kind "character" options review (default True). Note: the preview key's palette is S's
  own; the final keys share one palette over all directions, so S's colours can shift slightly.
  Tests tests/test_preview.py; shot Pictures/PixelPipeline/character_review_ui.png (tools/review_shot.py,
  knight-v3 copy, no GPU). GPU e2e (QA 2026-10-15, dwarf-front-first, gpu_lock): model 3.2 min, front + idle 49 s,
  redraw 37 s, pick try 1, approve 13.9 min (25 drawings 11.6 min + bending 1.8 min; S files sha1-identical);
  shot character_review_ui_gpu.png (two tries).
- **Backs + one colour per part (2026-10-15, Kevin: "their backs seem muddled and the green girl has a
  lump on her forehead")** (drawnrig.py, all drawn characters, runs after the reference S is drawn, so
  it fits "Front first, then OK"): cause = the Hunyuan model's far side is the front picture blurred
  through (mush), and every direction is repainted from ITS init, so a part the model has white (elf
  hair, sash) but S drew red came out white in some views (the "lump") and grey blobs on the back.
  (1) `reference_parts`: S's colours k-means'd (8, Lab, L half weight) and projected on the model; every
  point S sees takes its pixel's part + colour, unseen points the part of one of the 24 nearest seen
  points of their bone group (PART_GROUPS head/torso/arm/hand/leg/foot/tail) with the most similar
  MODEL colour, coloured from that part's S colours by lightness rank; part = group x cluster. The
  recoloured model (body.colours, the views' colours, the init cloud via nearest body point) makes
  every init agree with S. (2) BACK PICTURE (`draw_back`, files keys/back_N_t<i>.png, init_back_N.png):
  img2img 0.62 of the recoloured model seen from N at ~1024 px, model3d.view_prompt "back" (no pixel
  LoRA), up to BACK_PICTURE_TRIES=2 seeds, checked with subjectcheck.assess + part colours vs S
  (miss/grey) + IoU + sharpness; cut out, `paint_points` (z-buffer, facing) repaints the model's back,
  `relabel_by_colour` puts the painted points in their nearest S part, `match_points` evens them; on
  the frame grid it is N's reference panel: N painted beside [S | BACK] (view_refs), back diagonals
  beside [N | side] as before. Only drawn when a back view needs drawing (else a kept file is used).
  (3) every later drawing: `match_drawing` (part map = view_parts of its rest view): a part whose median
  colour is > PART_MISS 18 from S's median for it is recoloured whole by lightness rank to S's colours
  (shading kept), else only pixels > 24 from its part's AND every S colour; score -= miss + grey share
  (near-grey L>30 where S's part is coloured); GOOD also needs miss <= 0.15, grey <= 0.10. (4) back
  views still off (BACK_OK miss/grey/sharpness 0.6 x S) get BACK_RETRY (k0.6+2, o0.66+3) before the
  best is kept. drawn_sheets: redraw of >= half the directions makes a new palette (was: the old
  one, full of the white/grey it was fixing); back_*.png go to keys/old with N / a full redraw.
  info/keys.json: "drawings" (count this run), "parts" {model_recoloured, parts, back_picture{...}}.
  Tests (fake ComfyUI): test_drawnrig parts/back tests + updated counts (test_preview, test_robust use
  keys.json "drawings"). GPU (QA 2026-10-15, redraw of all but the approved S, gpu_lock per drawing):
  elf-archer 18.9 min (44 drawings incl. 1 back picture + retries, bending 94 s), dwarf-front-first
  12.1 min (26) and again 18.3 min (42, 5 back views retried). Shots Pictures/PixelPipeline/
  backs_fix_elf.png, backs_fix_2.png (dwarf, the 18.3 min run; backs_fix_2_v1.png the first), before
  over after, 16 idle + walk N|S. Result: backs read as a cloaked figure from behind in S's colours (elf:
  green hood + dark cloak instead of grey/white/green mush; dwarf: hood + cloak instead of his face and
  beard drawn on the back), no white hair. Tried and dropped (GPU): treating edge-on/outline points as
  unseen + nearest-only/up-facing parts for the far side (dwarf back views drew a face, elf bald),
  shot backs_fix_2_v2_rejected.png. Still weak: the elf's S itself is odd (brown hair + green top, a
  grey floating sash beside the head -> a grey puff by the head in SE/SSE, consistent with S but
  still a lump); dwarf NNW still drew a front (face) in the 2nd run, NE/ENE a pale beard edge
  (redraw one direction fixes such); dwarf backs dark-ish; the elf model's giant ribbon arcs.
  Side pictures (left/right) not made (E/W beside S were fine).
- **Progress bars (2026-10-15, Kevin: "can I get progress bars whenever doing this")** (`progress.py`,
  `comfyws.py`, ui/common.py ProgressPanel / GlobalProgress / JobRunner / hub()): one `progress.Tracker`
  per job IS the job's callback: text (as before; stage words by regex, "n of N", "about X min left"
  read), (step, text) (model3d), (done, total), a fraction; `child(k, n)` = batch job k of N. Plans =
  `Stage` lists in makeany.GPU minutes (`character_plan`, `model_plan`, `variations_plan`,
  `game_model_plan`, `makeany.progress_plan(plan)`); the first stage counts only if it happened;
  skipped stages drop out; bars never go back; ETA = plan x measured speed of finished stages, live
  "n of N" rate, hints. GPU steps: ComfyClient opens ComfyUI's /ws?clientId= (stdlib socket thread,
  only while a tracked job runs on the thread: `progress.using`; PySide's QtWebSockets is excluded by
  the spec) and wait() reports the prompt's sampler value/max (`gpu_sampling`, `gpu_pictures` i of n
  per call); no websocket = time-based. Stop: tracker.stop() -> the next say raises `progress.Stopped`,
  GPU waits POST /interrupt (stop_raises=False for batches: "after this one"). UI: ToolTab.run(...,
  plan=, estimate=, title=) puts a panel (overall bar + stage, stage/current-job bar + GPU step, about X
  min left, Stop) at the bottom of every page (attach_panel), ModelStudio/CharacterReview/MakeTab/
  objects/tiles/portraits/cards/Library export + restyle + queue pass plans; the main window's status
  bar shows the running job from any tab and the window title gets "NN% · ". Style training feeds the
  same hub. Tests tests/test_progress.py (fake clock, fake ComfyUI websocket); shot
  tools/progress_shot.py -> Pictures/PixelPipeline/progress_ui.png.
- **Claude drives the studio (2026-10-16, Kevin: "use Claude to build games and use this app as a
  tool")**: guide docs/FOR_CLAUDE.md (MCP config for a game repo, workflow, timings, gotchas) + a
  copyable skill docs/skill/pixel-pipeline/SKILL.md. `agent.py` = every app operation as a function
  returning JSON dicts (projects/house style, library list/get/tag/delete->.trash, plan/make for all
  12 makeany kinds, front-first character review/approve/redraw/words/pick/finish, animations/mocap/
  springs, reskin/restyle, game 3D model, map_info/map_edit, export, jobs); asset ids "kind/folder";
  previews under <project>/.agent/previews (directions grid, animations grid, variations, gif);
  `quality()` = subject check of the chosen picture + generator checks + drawn keys_report IoU/colour
  + empty frames -> warnings. `agentjobs.py`: job records <project>/agent_jobs/<id>.json, one GPU
  worker thread + cross-process O_EXCL lock (PIXEL_GPU_LOCK, default <home>/gpu.lock, same protocol
  as tools/gpu_lock), PIXEL_AGENT_PAUSE rest (20 s), CPU jobs on their own threads, Stop via the
  progress.Tracker, a dead owner pid -> "interrupted", resume() only on request (no unattended
  restarts). Jobs run under `assets.using_project` (thread-local current project, new, additive).
  Tilesets keep texture 1; the Made is pickled per job so pick_variation(job=) works after a restart.
  `gameexport.py`: export_to_game(engine godot|unity, project_path, subdir) -> per-kind writers
  (exports presets, uikit/cards/maplib exporters, tileset files), Godot pixel .import per PNG (not
  BMFont pages: the .fnt importer claims them), nested project.godot from map exports removed + res://
  re-rooted, project.godot Nearest filter, manifest pixel_pipeline_manifest.json (files/res paths,
  frame, animations {directions, frames, ms, loop}, feet anchor, ppu, tile, by_name), fingerprint
  (files size+mtime + options) -> incremental; verify_godot = --import (twice if errors) + load
  every manifest resource (Godot 4.6.1: 7/7 incl. character, UI theme, map, weather, water, parallax).
  MCP (mcp_server.py "asset studio" section): 47 new tools (17 old kept; the studio's tileset maker
  is make_terrain_tiles since make_tileset exists); CLI `cli_agent.py`: make/character/jobs/library/
  project/export with --json. Tests tests/test_agent.py (fake ComfyUI, robot draft, temp Godot/Unity
  projects, stdio).
- **HD-2D autoplay (2026-10-16, Kevin: "a video of fake gameplay, one iso and one top-down")**
  (`hd2d_autoplay.py`, make_hd2d(autoplay=dict|json, ui_kit=folder, unit_ppu={name: ppu})): a deterministic
  timeline in res://autoplay.json run by res://autoplay.gd when the game starts with `--autoplay` (or
  "auto": true); `--autoplay-quit` quits at the end (Movie Maker), `--autoplay-log` prints steps. board.gd:
  `scripted` (no input, no auto turn pass, `action_done` signal after walk/attack/finish), cam_pitch /
  cam_turn_rate / number_lift vars, unit ids (name, name#2..), prop_nodes per cell, bush/pot/cave props,
  play_effect(tag=, scale=, flip=) returns the fx (+stop_effect), effect.json "tags" from frameTags, effects
  on layer 2 out of spell/effect lights (no self-lit white-outs), _attack(dmg=, fx=), sprite shader lean +
  flip_x. Modes: tactics (select/range/move/attack/cast/heal/end_turn banners, UI-kit unit panel + turn
  plate) and action (goto waypoints 8-way, swing+knockback, hurt+blink, die+poof+drops, cut bushes,
  spawn/pickup hearts/coins/gems, ALttP HUD from the kit). Steps documented in the module docstring;
  points are cell coords ([c, r] = centre); "when" = wait for the clock. Tests tests/test_hd2d_autoplay.py
  (headless runs twice, same HP/time). Sample: Pictures/PixelPipeline/hd2d_gameplay/{iso,action} (+ _make/
  build.py render.py encode.py; QA v3 tiles, QA/Neo16 characters + re-skins) -> hd2d_gameplay_iso.mp4 (48 s)
  / hd2d_gameplay_action.mp4 (31 s) + _small.mp4 (720p). Movie Maker 1080p: ~70 s per 48 s clip.
- **Playtests (2026-10-16, Kevin: "a tiny map for each style" he can PLAY to point out issues; "game play in
  the long run")** (`playtest.py`, `godot_runtime/`): `build_playtest(style, out_dir, Sources)` / `build_all(root)`
  for tactics | action | platformer | iso | cards -> Pictures/PixelPipeline/playtest/<style>/ + Play_/Edit_<style>.bat
  + README.txt. Built through the app's own paths: make_hd2d (tactics = real3d.tscn main scene, 10x8 river crossing;
  iso = res://iso/iso.tscn of the same map; action = play_action.tscn running godot_runtime/action_play.gd, which
  EXTENDS res://autoplay.gd and reuses its HUD/swing/hit/die/drops/pickups/projectiles with live input), and
  gameexport.export_to_game + manifest into fresh Godot projects (platformer: side-scroller staging project, tilegen
  side set (terrain painting + one-way platforms), backgrounds parallax, effects, UI kit, hero re-rendered E/W + jump by
  drawnrig.drawn_sheets(directions=2) bending the EXISTING keys on the CPU (a _NoGPU client makes any draw request
  fail -> falls back to the 16-dir sheet); cards: 17 cards made with cards.py on the CPU (set_art from characters'
  painted pictures, items, a prop, effect peak frames, the QA Fire card), exported Card.tscn per card, enemies = library
  characters + reskin.py re-skins staged as characters). Sources are read only; staging projects live in
  <root>/_studio/projects (not the user's project list). Runtime (package data `godot_runtime/*.gd`, spec DATAS):
  pp_debug.gd autoload PPDebug (F1 hint, F3 overlay = FPS + debug_state() of nodes in group "pp_debug" + 2D collision
  shapes / debug_draw(), F12 report -> res://reports/<style>_<time>.png/.txt with the last 3 s of state + inputs, F5
  restart; `--pp-selftest` runs selftest() coroutines -> "PPTEST {json}", `--pp-shot=<png>` shot_setup() + screenshot),
  pp_tactics.gd autoload PPTactics (AI for team != 0 on board.gd via `scripted` + action_done, and on iso.gd via its
  select/move_selected/attack API; blocks input while it plays), side_game/side_player/side_enemy/pickup_2d (platformer:
  coyote + buffer + variable jump, stomp, arrows, pits), card_game.gd (map fight -> fight|rest|shop -> elite|rest ->
  boss, draw/discard/exhaust, energy, block, strength, poison/burn, intents, pick 1 of 3, deck viewer, shop/remove),
  pp_manifest.gd (manifest by id/name). All five pass headless selftests (AI turns, movement, a whole card run).
  Tests tests/test_playtest.py. Known then (iso scale, cyan blobs, pond glare: fixed by the game runtime below): iso sprites at native sheet size (knight 144x96
  vs elf 96x64 look different sizes), side hero drawn at the keys' 30 deg (true side view needs GPU drawings), soft cyan
  blobs float above the blue units on the tactics board (cause not checked), pond glare at noon in the action map.
- **Game runtime: clipping + footsteps + smooth motion (2026-10-17, Kevin: "bad clipping, footsteps
  don't work well, looks very clunky"; gameplay matters long term)**. One runtime, used by the HD-2D
  board, autoplay, the playtests and shipped with every export (`gameexport.install_runtime` ->
  `<subdir>/pp_runtime/` + manifest "runtime"/"motion"): `godot_runtime/pp_motion.gd` (static: approach
  accel/decel, path_speed, turn at 900 deg/s through the 16 facings, locomotion = walk/run by speed at
  speed/stride cycles/s with cadence caps, settle to the next contact pose, hop_y, ground_follow,
  smooth_damp camera, shake_offset, hit_time), `pp_sprite3d.gdshader` (= hd2d.SPRITE_SHADER: every
  pixel takes the FEET's depth + depth_bias 0.35 toward the camera -> no cutting into ground/walls/
  stairs/props/each other, still hidden by what's in front; whole-screen-pixel snap), `motion.json`
  (`motion.MOTION`; Unity gets the json + the algorithms in docs/FOR_CLAUDE.md). board.gd: units have
  yaw_target/phase/rate/speed/mover; `move_along` (rounded corners, eased, hops on level changes via
  ground_at), `_walk` uses it; attacks = anticipation lean, blow on the clip's strike frame
  (`hit_time`), hit-stop (`freeze`), knockback offset, trauma shake, follow-through; damped camera
  (`cam_target`, autoplay/pp_tactics use it); HP bars + damage numbers in 2D (the floating "soft cyan
  blobs" were the 3D Sprite3D HP bars (blue frame + green fill) blurred by depth of field); props in
  map.json "blocking" (diorama.BLOCKING_PROPS + bush/pot: trees/houses/wells no longer walk-through)
  + `obstacles` radii; `apron` (scenery cells past the map edge, make_hd2d(apron=)); water light()
  with capped banded glare ("water_spec"/"glare" per preset). action_play.gd: velocity hero, circle
  collision vs cell boxes + prop circles with push-out sliding, foes the same. Strides: `motion.py`
  (rig_stride: planted ankle/paw speed x cycle; measure_sheet rig|frames|default; px_per_unit = keys
  camera scale or idle height) -> sheet json meta.stride/body_px (library.annotate_stride after every
  render), unit.json stride (tiles), manifest animations[clip].stride + px_scale/world_ppu/body_px.
  Footsteps at the source: autorig walk/run = planted-feet 2-bone IK gait (`_Moves.gait`, GAIT reach/
  duty; the old sine swing slid the feet), quadruped strides 0.2/0.36 -> 0.34/0.5 leg; per-clip frame
  counts everywhere (anim3d.frames_of / dict frames, export.pack_rows, normals rows=, AnimatedSet.
  counts), `Style.locomotion_frames` 12 (new characters; old sheets keep their counts: library.
  frames_for), `library.rerender_locomotion(ch, 12|16)` (CPU only: `drawnrig.no_new_drawings()` makes
  any GPU request skip/raise NeedsDrawing - without it the quality loop RETRIED a muddled back view on
  the GPU). Iso: `rescale_sheet`/`downscale_pixel_art` + make_hd2d(iso_scale=) put all units on one
  scale; per-unit step_time/walk_scale from stride. Platformer hero: speed_scale from stride. Cards:
  sprites at whole scales by px_scale + the hero's painted portrait. Playtests: `locomotion_copies`
  (staged 12-frame planted-feet copies, `_studio/projects/loco-*`). Tests: tests/test_runtime_motion.py.
- The container that built this couldn't reach dl.google.com, so APKs build only in
  GitHub Actions.

## File map

- `settings.py`: `Settings` + per-subject `PRESETS`; `settings_for()` (None overrides =
  keep preset; height 0 = auto aspect)
- `pipeline.py`: load → isolate → prepare (crop, fit, fill, pixelize) → finish (palette,
  dither, cleanup, outline); `convert`, `convert_batch`, `sprites_from_rgba` (shared
  palette)
- `tilegen.py` + `tilepaint.py` (tiles v3: painted stones, drawn soft ground, fold, stone_tile, boulder_face) +
  `tilewords.py` (words -> drawn ground shapes)
- `pixelize.py` (outline expansion, contrast downscale, tileable), `quantize.py`,
  `palettes.py`, `finishing.py` (orphan cleanup, sel-out), `color.py`
- `video.py` (frames → aligned sprites, track/fixed), `animate.py` (breathe/bob/sway)
- `export.py` (sheets + JSON, tilesets + Tiled .tmj, `write_animation`, `load_sprite`
  undoes upscaled exports), `aseprite.py`, `godot.py`
- `cli.py` (convert, video, idle, sheet, tileset, palettes), `ui/app.py` (desktop),
  `mcp_server.py` (MCP tools for Claude; needs mcp>=2), `agent.py` / `agentjobs.py` /
  `gameexport.py` / `cli_agent.py` (Claude drives the studio: docs/FOR_CLAUDE.md)
- `systems.py` (retro looks), `drawing.py` (text-row sprites), `generate.py` (ComfyUI)
- `depth.py` + `parallax.py`: depth-based parallax layers + Godot Parallax2D scene
- `packaging/` (PyInstaller spec), `.github/workflows/ci.yml` (tests on win+linux, exe on
  main), `.github/workflows/android.yml` (APK + `android-latest` release)

## Conventions

- Images are numpy uint8 RGBA (H, W, 4); sprite alpha is binary (0/255).
- Perceptual maths in Lab via `color.py`.
- Run `pytest` before pushing; the UI test runs headless (`QT_QPA_PLATFORM=offscreen`).
  Faster: `python tools/run_tests.py [--jobs 4] [-- pytest args]` = several pytest processes over
  file groups balanced by tests/file_times.json (`--update-times` refreshes it; CI runs
  `--split 2 --part N`, 2 jobs per OS). conftest sets PIXEL_KD_WORKERS=-1 (drawnrig's cKDTree
  queries on every core, same answers; app default 1): those searches were ~60% of the
  drawn-character tests (2026-10-16: serial suite 36 -> ~24 min). Tests run at idle priority
  (conftest; PIXEL_TEST_PRIORITY=normal off); the runner defaults to <= 3 processes.
- Keep workflows lean (Actions minutes are free while the repo is public).

## Repo and builds

GitHub: `sparkskevin96-cell/pixel-pipeline` (created 2026-10-10 private, made **public**
by Kevin 2026-10-11 so the app updates itself with no token). Pushes to `main`
run CI (tests on Linux + Windows) and publish the Windows app as the `windows-latest`
pre-release; changes under `mobile/` or `web/` publish the APK as `android-latest`. The
Claude GitHub app can't dispatch workflows (403), so trigger by pushing.

**Install + self-update** (`ui/updater.py`, windows in `ui/update_ui.py`), 2026-10-10, after
Kevin opened an old copy: CI writes `build.json` (run number, sha, date) into the app and
uploads it as `version.json` on the release. A downloaded copy offers to install itself to
`%LOCALAPPDATA%\Programs\PixelPipeline` (+ desktop/Start menu shortcuts) or opens the
installed copy if that's as new. Updating = download zip → unpack to
`%LOCALAPPDATA%\PixelPipeline\update` → PowerShell script waits for the pid, robocopies
(`_internal` mirrored) and restarts with `--updated`. Public repo => no token. If it is ever made
private again, the check needs a fine-grained token (Contents: read) saved as "github" in
keys.json (Help → Updates token). CI's "Update test" step updates a build-0 copy from the real zip; the publish step
checks the release through the API.

**Version number** = the CI run number of ci.yml (`build.json`), shown bottom-right in the
app, in its title and in the release title. Kevin asked (2026-10-10): after every push that
rebuilds the Windows app, wait for CI and tell him the new version number. Docs-only pushes
don't rebuild (paths-ignore), so no new version then.

Before the repo existed (2026-10-07..09), builds ran from branch `ccr-15172514-6ugcr0` of
`sparkskevin96-cell/pawgress` (local branch `pawgress-build`), with pre-releases
`pixel-pipeline-android-temp` and `pixel-pipeline-windows-temp` there. Cleaned up
2026-10-10 with Kevin's OK: both pre-releases and tags deleted, and that branch reset to
Pawgress `main` (this session can't delete branches).

## Next up

AI style pass (PixelLab / Retro Diffusion / ComfyUI), palette ramps and shading, batch
tools in the desktop app, video import on phone.
