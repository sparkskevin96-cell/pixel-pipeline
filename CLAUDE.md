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
- **Isometric tactics look** (`iso_scene.py`, CLI `iso-demo`), 2026-10-11, one of the three looks
  of the tactics demo over `diorama.py`: `write_iso_scene(d, project, {name: sheet.png})` writes
  `iso/` (+ map/map.json if absent, project.godot only if absent). One pre-rendered column per
  cell (top + sides from its height down to level -2; tile W, level = W/4 px; side texture 1:1,
  line per level), painter's order by z_index = (col+row)*8 + layer. GDScript range = path_cost
  (test compares them in Godot). Grid +col faces SE, +row SW. Verified on Godot 4.6.1 with real
  GPU screenshots (`Pictures/PixelPipeline/hd2d_iso_test/shot_*.png`, `shot.gd` makes them).
- The container that built this couldn't reach dl.google.com, so APKs build only in
  GitHub Actions.

## File map

- `settings.py`: `Settings` + per-subject `PRESETS`; `settings_for()` (None overrides =
  keep preset; height 0 = auto aspect)
- `pipeline.py`: load → isolate → prepare (crop, fit, fill, pixelize) → finish (palette,
  dither, cleanup, outline); `convert`, `convert_batch`, `sprites_from_rgba` (shared
  palette)
- `pixelize.py` (outline expansion, contrast downscale, tileable), `quantize.py`,
  `palettes.py`, `finishing.py` (orphan cleanup, sel-out), `color.py`
- `video.py` (frames → aligned sprites, track/fixed), `animate.py` (breathe/bob/sway)
- `export.py` (sheets + JSON, tilesets + Tiled .tmj, `write_animation`, `load_sprite`
  undoes upscaled exports), `aseprite.py`, `godot.py`
- `cli.py` (convert, video, idle, sheet, tileset, palettes), `ui/app.py` (desktop),
  `mcp_server.py` (MCP tools for Claude; needs mcp>=2)
- `systems.py` (retro looks), `drawing.py` (text-row sprites), `generate.py` (ComfyUI)
- `depth.py` + `parallax.py`: depth-based parallax layers + Godot Parallax2D scene
- `packaging/` (PyInstaller spec), `.github/workflows/ci.yml` (tests on win+linux, exe on
  main), `.github/workflows/android.yml` (APK + `android-latest` release)

## Conventions

- Images are numpy uint8 RGBA (H, W, 4); sprite alpha is binary (0/255).
- Perceptual maths in Lab via `color.py`.
- Run `pytest` before pushing; the UI test runs headless (`QT_QPA_PLATFORM=offscreen`).
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
