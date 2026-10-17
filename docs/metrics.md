# How long making a character takes

Kevin (2026-10-17): "how long to create a character: currently X, after Y".

Every character creation now records its stage times in the character (`asset.json`
`extra.timing`, mirrored into `keys/keys.json` `"timing"`), shown in the Character Library
(tooltip + status line), on the front-first review panel ("So far: N min") and in the agent/MCP
asset summary (`timing`). The time the app waits for your OK on the front drawing is NOT a stage:
`wall` = the sum of the stages = the time the app worked on it. Aggregate over the last N
characters: `pixel-pipeline metrics [-n 10] [--json]`, agent `metrics()`, MCP `character_metrics`.

Stages: `model` (picture + 3D shape), `rig` (skeleton + clips), `front` (front drawing +
preview idle), `drawings` (the other directions), `quality_check` (bending idle + walk for the
check, measuring every direction), `quality_redraw` (failing directions drawn again),
`bending` (every animation bent from the drawings, clean-up included), `sheets` (3D-rendered
characters only).

## X: before (measured runs, RTX 4070 laptop 8 GB, 64 px people)

| Character (date) | model | front | other directions | bending | total |
|---|---|---|---|---|---|
| dwarf-front-first (2026-10-15) | 3.2 | 0.8 | 11.6 (25 drawings) | 1.8 | ~18 min |
| elf-archer, backs fix (2026-10-15/16) | 3.2 | 0.8 | 18.9 (44 drawings incl. back picture + retries) | 1.6 | ~24.5 min |
| dwarf rerun, backs fix | 3.2 | 0.8 | 18.3 (42 drawings) | ~1.5 | ~24 min |

**X = about 18-25 min per character** (GPU-bound; excluding the review wait). No per-direction
quality check: weak directions shipped as they came out.

## Y: after (automatic pipeline, 128 px people = 2x, per-direction check + bounded redraws)

See the table at the end (filled from the 2x test runs in project "Pixel 2x test").

### Measured 2026-10-17 (project "Pixel 2x test": 128 px people, 64 px tiles, 45 deg; minutes)

The 3D model of each was REUSED from the 1x character (its measured 3.2 min is counted as `model`).

| stage | knight-2x 45 deg (first run, before the cuts) | **knight-2x30 (30 deg, after the cuts: the clean Y)** | elf-2x (after cuts*) | wolf-2x (after cuts) | side hero (2 dirs) |
|---|---|---|---|---|---|
| model (picture + shape) | 3.2 | 3.2 | 3.2 | 3.2 | 3.2 |
| rig | 0.2 | 0.04 | 0.05 | 0.04 | 0.01 |
| front + preview | 1.8 (incl. ~1.3 cold ComfyUI start) | 0.6 | 0.2* | 0.8 | 1.2 |
| other directions drawn | 18.0 (43 drawings + back picture) | 18.6 (38 drawings + back picture) | 14.5 | 9.8 | 1.0 |
| quality check (idle+walk bends, measuring) | 1.6 | 2.2 | 2.1 | 1.3 | 0.35 |
| quality redraws | 8.8 (4 back views x ~5 tries + a 2nd back picture) | 1.65 (SW, N: 1 try each) | 2.8 (4 directions, 1 try each) | 1.5 (3 directions) | 0.6 |
| bending every animation (+ clean-up) | 2.4 | 2.6 | 2.9 | 1.6 | 0.4 |
| **total (wall, review wait excluded)** | **36.0** (31.7 GPU) | **29.0** (24.1 GPU) | **25.7** (20.7 GPU)* | **18.2** (15.3 GPU) | 6.7 |
| directions passed (automatic check) | 8/16 at the end of the run; 13/16 after the clean-up re-bend | **15/16** (redrawn SW N; weak: W) | 8/16 (redrawn W NW WNW E) | **16/16** (redrawn SSW SE) | 0/2 (bow pieces count as torn bits) |

\* elf-2x: its first run was stopped after ~8 drawings to apply the cuts; the restart got those 8
pictures back from the picture cache (same workflow = same picture, ~3 min less GPU). A clean elf
run is therefore ~28-29 min.

Where the knight's 36 min went (log analysis): drawings at ~23 s each (median, GPU-bound); per
candidate kind: o0.6 8.9 min, k0.6 8.0, o0.72 5.3, **o0.66 3.5 (never won a direction in 9
characters / ~56 back views)**, back pictures 1.35 (two). No heat rests (GPU stayed under 80 C); the
only model load was the cold ComfyUI start in the front stage.

Cuts made (no quality loss found):
1. Quality redraws draw ONE candidate per direction (the one that wins most: o0.6 front, k0.6 back),
   not the full set + retries: knight 20 redraw pictures -> ~5.
2. A quality redraw of N keeps the checked back picture (no second back picture + model switch).
3. The back retry `o0.66+3` is gone (0 wins ever); `k0.6+2` stays (won 4 of ~56 backs).
4. The final full bend draws nothing new (no_new_drawings) - only the check bends idle+walk first.

Kept on purpose (cutting would cost quality): the front's second try o0.72 (wins 26% of front
views), the back views' o0.6 / o0.72 tries (23% / 11% of backs), drawing at ~1024 px (the 2x frame
keeps more of it; smaller drawings lose the detail Kevin asked for), mirroring only when the two
sides really match (symmetric_pair: the knight, elf and wolf don't).

**Y (after): 18-29 min per 2x character (wolf 18.2, knight 29.0, elf ~28),
vs X = 18-25 min for 1x characters.** The 2x size itself costs almost nothing on the GPU (the
drawings are ~1024 px either way); what it adds is CPU bending (~1.5-3 min vs ~1.5) and the
automatic check (~1.3-2 min, CPU) + its bounded redraws (~1.5-3 min GPU). In exchange every
direction is measured, the worst ones are redrawn without anyone asking, and the result says which
directions are still weak.

Why the knight was redrawn at 30 deg: drawn at 45 deg its standing body came out only ~86 px of
the 128 px frame (seen from higher up a standing figure is shorter, and the frame must also fit
the lying death pose), so in the game it was SMALLER than the old 96 px knight. Drawn at 30 deg
(the HD-2D default) the body uses ~110 px. The game camera's own angle no longer has to match:
the board picks the drawing whose on-screen facing matches (`_sprite_dir`).

Knight 1x vs 2x: X for the 1x knight-v3 was 12.5 min of drawing (27 drawings) + ~2 min bending +
model 3.2 = ~18 min, so the 2x knight with the check costs ~11 min more: ~4 min of it are the
extra drawings this run needed (38 vs 27: more back views were muddled and retried), ~2 min the
check bends, ~1.7 min its redraws, ~0.6 min 2x bending.

## Fixes round (2026-10-17): elf remade with the 3D model checks

Measured on the RTX 4070 laptop, project "Pixel 2x test" (128 px people, 45 deg), everything
automatic (picture check, shape check + one bounded shape retry, per-direction check + bounded
redraws), inside the shared GPU lock. Minutes.

| stage | elf-2x (before, X) | **elf-2x-b (after, Y)** | elf-side-2x (before) | **elf-side-2x-b (after)** |
|---|---|---|---|---|
| model (picture + shape + check) | 3.2 (reused) | **3.8** (1 picture, 2 shapes: the first was 54% flyaways, a new seed came out clean) | 3.2 (reused) | 3.8 (the elf-2x-b model, reused) |
| rig | 0.05 | 0.07 | 0.01 | 0.04 |
| front + other directions drawn | 14.7 | 14.5 (36 drawings + back picture) | 2.2 | 1.5 |
| quality check | 2.1 | 2.0 | 0.35 | 0.2 |
| quality redraws | 2.8 (4 directions) | 1.4 (2 directions) | 0.6 | 0 |
| bending | 2.9 | 2.7 | 0.4 | 0.24 |
| **total (wall)** | **~28** (25.7 + restart) | **24.5** | 6.7 | **5.8** |
| directions passed | 8/16 (floating bow / ribbon bits) | **15/16** (weak: SW blotch) | 0/2 | **2/2** |

**X -> Y: elf ~28 min -> 24.5 min**, 8/16 -> 15/16 directions; the side elf 6.7 -> 5.8 min, 0/2 -> 2/2.
The model checks cost ~2 s per shape on the CPU; the bounded shape retry ~1.3 min of GPU only
when a shape is mostly flyaways (it was, once). Fewer broken directions = fewer redraws, so the
whole character came out faster.

Not creations, but the same automatic tools on existing characters (CPU re-bend + check, then
`check_and_fix` with bounded GPU redraws): knight-2x30 relit backs (its old skeleton kept: a new one
under old drawings changed the front views) 7.7 min check_and_fix -> 14/16 (W passes; weak: N sharp,
ENE flicker); wolf-2x at its real size
(drawings re-placed, no GPU) 1.0 min + 6.6 min check_and_fix (2 redraws) -> 15/16.
