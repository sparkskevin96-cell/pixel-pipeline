# Sprite scale by genre

How tall the main character is as a share of screen height in classic 2D games, and the sprite height that keeps that share in a 16:9 game. Interactive version: https://claude.ai/artifact/Jv1VrzgNjgmiH4LgAQg1SR (private to Kevin).

Heights are the standing body in game pixels (no sideways weapons, shadows or HUD). NES/SNES/Genesis/CPS1 use the 224 visible NTSC lines, GBA 160, Game Boy 144. Targets are the median of the classic games in each row; modern games are reference only. Kevin's games render at 1080p (4K is exactly 2x), so any base that divides 1080 works: 320x180 (x6), 384x216 (x5), 480x270 (x4), 640x360 (x3). Sprite height = share x base height. Photo conversion stays recognizable from about 48 px tall (our own astronaut tests).

| Genre | Classic range | Median | 320x180 | 640x360 |
|---|---|---|---|---|
| Fighting | 40.6-50.8% | 46.9% | 84 px | 169 px |
| Beat 'em up | 23.2-41.5% | 35.3% | 63 px | 127 px |
| RPG battle, close-up | 31.3-38.2% | 34.4% | 62 px | 124 px |
| Tactics battle scene | 13.8-37.9% | 14.5% | 26 px | 52 px |
| Metroidvania | 13.4-20.0% | 16.3% | 29 px | 59 px |
| Run-and-gun | 15.2-17.9% | 16.5% | 30 px | 59 px |
| Platformer | 10.7-17.4% | 13.6% | 25 px | 49 px |
| Top-down action-adventure | 7.1-14.3% | 11.8% | 21 px | 42 px |
| RPG field / overworld | 7.1-19.4% | 10.7% | 19 px | 39 px |
| RPG battle, side view | 10.7-10.7% | 10.7% | 19 px | 39 px |
| Farming / life sim | 10.3-10.3% | 10.3% | 18 px | 37 px |
| Tactics map | 7.1-16.9% | 10.0% | 18 px | 36 px |
| Shoot 'em up (ship) | 4.7-7.1% | 5.9% | 11 px | 21 px |

GBA Fire Emblem battle sprites (22.5%), the look Kevin asked about first: 40 px at 320x180, 81 px at 640x360.

## Every game

| Genre | Game | System | Screen | Char px | Share | Source |
|---|---|---|---|---|---|---|
| Fighting | Street Fighter II (Ryu) | Arcade CPS1 | 224 | 91 | 40.6% | [Ryu sheet + frame rects](https://github.com/Mayank-Jain-1/StreetFighter), measured |
| Fighting | The King of Fighters '98 | Neo Geo | 224 | 105 | 46.9% | [Native screenshot](https://github.com/libretro-thumbnails/SNK_-_Neo_Geo), measured |
| Fighting | Mortal Kombat (1 and 3) | Arcade | 254 | 129 | 50.8% | [MK3 sprite rips](https://github.com/mgechev/mk.js), measured |
| Beat 'em up | TMNT IV: Turtles in Time | Super NES | 224 | 52 | 23.2% | [Native screenshot](https://github.com/libretro-thumbnails/Nintendo_-_Super_Nintendo_Entertainment_System), measured |
| Beat 'em up | Streets of Rage 2 (Axel) | Genesis | 224 | 79 | 35.3% | [Native screenshot](https://github.com/libretro-thumbnails/Sega_-_Mega_Drive_-_Genesis), measured |
| Beat 'em up | Final Fight (Cody) | Arcade CPS1 | 224 | 93 | 41.5% | [Native screenshot](https://github.com/libretro-thumbnails/MAME), measured |
| RPG battle, close-up | Golden Sun (battle) | GBA | 160 | 50 | 31.3% | [Golden Sun engine port](https://github.com/jjppof/goldensun_html5), measured |
| RPG battle, close-up | Pokémon Emerald (trainer) | GBA | 160 | 55 | 34.4% | [pokeemerald decompilation](https://github.com/pret/pokeemerald), measured |
| RPG battle, close-up | Pokémon Red/Blue (trainer) | Game Boy | 144 | 55 | 38.2% | [pokered disassembly](https://github.com/pret/pokered), measured |
| Tactics battle scene | Fire Emblem (NES) | Famicom | 224 | 31 | 13.8% | [Fire Emblem Wiki sprite sizes](https://fireemblemwiki.org/wiki/Category:Genealogy_of_the_Holy_War_battle_sprites), documented |
| Tactics battle scene | Fire Emblem Gaiden | Famicom | 224 | 33 | 14.7% | [Fire Emblem Wiki sprite sizes](https://fireemblemwiki.org/wiki/Category:Genealogy_of_the_Holy_War_battle_sprites), documented |
| Tactics battle scene | FE: Mystery of the Emblem | Super Famicom | 224 | 32 | 14.3% | [Fire Emblem Wiki sprite sizes](https://fireemblemwiki.org/wiki/Category:Genealogy_of_the_Holy_War_battle_sprites), documented |
| Tactics battle scene | FE: Genealogy of the Holy War | Super Famicom | 224 | 32 | 14.3% | [Fire Emblem Wiki sprite sizes](https://fireemblemwiki.org/wiki/Category:Genealogy_of_the_Holy_War_battle_sprites), documented |
| Tactics battle scene | FE: Thracia 776 | Super Famicom | 224 | 31 | 13.8% | [Fire Emblem Wiki sprite sizes](https://fireemblemwiki.org/wiki/Category:Genealogy_of_the_Holy_War_battle_sprites), documented |
| Tactics battle scene | Fire Emblem 6–8 | GBA | 160 | 36 | 22.5% | [FE8 decompilation](https://github.com/FireEmblemUniverse/fireemblem8u), measured |
| Tactics battle scene | Advance Wars (battle) | GBA | 160 | 28 | 17.5% | [Sprite data](https://github.com/XoanOuteiro/advance-wars-colorscripts), measured |
| Tactics battle scene | Shining Force (battle) | Genesis | 224 | 85 | 37.9% | [Shining Force disassembly](https://github.com/ShiningForceCentral/SF1DISASM), estimate |
| Metroidvania | Castlevania | NES | 224 | 30 | 13.4% | [Sprite rip](https://github.com/KutieKat/Castlevania), estimate |
| Metroidvania | Castlevania III | NES | 224 | 30 | 13.4% | [Native screenshot](https://github.com/cyneprepou4uk/NES-Games-Disassembly), measured |
| Metroidvania | Super Metroid | Super NES | 224 | 43 | 19.2% | [ROM-rendered capture](https://github.com/TooManyBees/super-metroid), measured |
| Metroidvania | Symphony of the Night | PlayStation | 240 | 48 | 20.0% | [SotN decompilation (hitbox)](https://github.com/Xeeynamo/sotn-decomp), estimate |
| Run-and-gun | Contra | NES | 224 | 34 | 15.2% | [Contra disassembly](https://github.com/vermiceli/nes-contra-us), measured |
| Run-and-gun | Metal Slug | Neo Geo | 224 | 40 | 17.9% | No source found, estimate |
| Platformer | Super Mario Bros. (Super) | NES | 224 | 32 | 14.3% | [Sprite rip](https://github.com/justinmeister/Mario-Level-1), measured |
| Platformer | Super Mario Bros. 3 (Super) | NES | 224 | 26 | 11.6% | [SMB3 disassembly](https://github.com/captainsouthbird/smb3), measured |
| Platformer | Super Mario World (Super) | Super NES | 224 | 29 | 12.9% | [SMW reimplementation (hitbox)](https://github.com/snesrev/smw), estimate |
| Platformer | Mega Man 2 | NES | 224 | 24 | 10.7% | [Sprite rip](https://github.com/MegaChibisX/MegaMan-Unity-8Bit-Engine), measured |
| Platformer | Mega Man X | Super NES | 224 | 34 | 15.2% | [Sprite rip](https://github.com/sharivan/XSharp), measured |
| Platformer | Sonic the Hedgehog | Genesis | 224 | 39 | 17.4% | [Sonic 1 disassembly](https://github.com/sonicretro/s1disasm), measured |
| Platformer | Celeste (modern) | PC, 320×180 | 180 | 13 | 7.2% | [Celeste source (hitbox)](https://github.com/NoelFB/Celeste), estimate |
| Platformer | Shovel Knight (modern) | PC, 400×240 | 240 | 32 | 13.3% | [Yacht Club blog (resolution)](https://www.yachtclubgames.com/blog/breaking-the-nes/), estimate |
| Top-down action-adventure | The Legend of Zelda | NES | 224 | 16 | 7.1% | [Native screenshot](https://github.com/libretro-thumbnails/Nintendo_-_Nintendo_Entertainment_System), measured |
| Top-down action-adventure | A Link to the Past | Super NES | 224 | 22 | 9.8% | [SpriteSomething Link sheet](https://github.com/Artheau/SpriteSomething), measured |
| Top-down action-adventure | The Minish Cap | GBA | 160 | 22 | 13.8% | [Native screenshot](https://github.com/libretro-thumbnails/Nintendo_-_Game_Boy_Advance), measured |
| Top-down action-adventure | Secret of Mana | Super NES | 224 | 32 | 14.3% | [Native screenshot](https://github.com/libretro-thumbnails/Nintendo_-_Super_Nintendo_Entertainment_System), measured |
| RPG field / overworld | Dragon Quest | NES | 224 | 16 | 7.1% | [Dragon Warrior disassembly](https://github.com/nmikstas/dragon-warrior-disassembly), documented |
| RPG field / overworld | Final Fantasy | NES | 224 | 16 | 7.1% | [FF1 disassembly](https://github.com/Entroper/FF1Disassembly), documented |
| RPG field / overworld | Final Fantasy IV | Super NES | 224 | 16 | 7.1% | [FF4 documentation](https://github.com/aexoden/ff4-documentation), documented |
| RPG field / overworld | Final Fantasy VI | Super NES | 224 | 24 | 10.7% | No source found, documented |
| RPG field / overworld | EarthBound | Super NES | 224 | 24 | 10.7% | [CoilSnake wiki](https://github.com/pk-hack/CoilSnake/wiki/Overworld-Sprites), documented |
| RPG field / overworld | Pokémon Red/Blue | Game Boy | 144 | 16 | 11.1% | [pokered disassembly](https://github.com/pret/pokered), measured |
| RPG field / overworld | Pokémon Emerald | GBA | 160 | 21 | 13.1% | [pokeemerald decompilation](https://github.com/pret/pokeemerald), measured |
| RPG field / overworld | Chrono Trigger | Super NES | 224 | 34 | 15.2% | [Native-scale rip](https://github.com/kirbysayshi/CT), measured |
| RPG field / overworld | Golden Sun | GBA | 160 | 31 | 19.4% | [Golden Sun engine port](https://github.com/jjppof/goldensun_html5), measured |
| RPG battle, side view | Final Fantasy (battle) | NES | 224 | 24 | 10.7% | [FF1 disassembly](https://github.com/Entroper/FF1Disassembly), measured |
| RPG battle, side view | Final Fantasy IV (battle) | Super NES | 224 | 24 | 10.7% | [FF4 documentation](https://github.com/aexoden/ff4-documentation), estimate |
| RPG battle, side view | Final Fantasy VI (battle) | Super NES | 224 | 24 | 10.7% | No source found, documented |
| Farming / life sim | Harvest Moon | Super NES | 224 | 23 | 10.3% | [Sprite rip](https://github.com/mimikim/Harvest-Moon-SNES-Feed-Jack-HTML5-Canvas), measured |
| Farming / life sim | Stardew Valley (100% zoom) (modern) | PC, 1080p | 1080 | 128 | 11.9% | [Stardew wiki](https://stardewvalleywiki.com/Modding:Farmer_sprite), documented |
| Tactics map | Fire Emblem 1–5 (map) | Famicom / SFC | 224 | 16 | 7.1% | [Fire Emblem Wiki sprite sizes](https://fireemblemwiki.org/wiki/Category:Genealogy_of_the_Holy_War_battle_sprites), documented |
| Tactics map | Fire Emblem 6–8 (map) | GBA | 160 | 16 | 10.0% | [FE8 decompilation](https://github.com/FireEmblemUniverse/fireemblem8u), measured |
| Tactics map | Advance Wars (map) | GBA | 160 | 15 | 9.4% | [Sprite data](https://github.com/XoanOuteiro/advance-wars-colorscripts), measured |
| Tactics map | Shining Force (map) | Genesis | 224 | 24 | 10.7% | [Shining Force disassembly](https://github.com/ShiningForceCentral/SF1DISASM), documented |
| Tactics map | FF Tactics Advance | GBA | 160 | 27 | 16.9% | [In-game capture](https://github.com/kttalley/ffta-decode-public), measured |
| Shoot 'em up (ship) | Gradius (ship) | Arcade | 224 | 16 | 7.1% | [Native screenshot](https://github.com/libretro-thumbnails/MAME), measured |
| Shoot 'em up (ship) | R-Type (ship) | Arcade | 256 | 12 | 4.7% | [Native screenshot](https://github.com/libretro-thumbnails/MAME), measured |
