# Pixel Pipeline for Android

The web app (`../web/index.html`, the photo → sprite converter in JS/canvas) in a
full-screen WebView, plus native features through a JavaScript bridge: camera and
photo picker, on-device subject cut-out (Google ML Kit), saving to the gallery and the
share sheet. Flutter, Android only. A personal sideload build, not a Play Store app.

## Install on your phone

Every push to `main` that touches `mobile/` or `web/` builds the APK (GitHub Actions,
`.github/workflows/android.yml`) and replaces the **android-latest** pre-release.

1. On the phone, open the repo on github.com (signed in, it's private) → **Releases**
   → **Android app 1.0.N** → tap **pixel-pipeline-arm64.apk**. The link never changes:
   `https://github.com/<owner>/pixel-pipeline/releases/download/android-latest/pixel-pipeline-arm64.apk`
2. Open the download. If Android asks, allow your browser to **install unknown apps**
   (Settings → Apps → Chrome → Install unknown apps), then go back and tap Install.
3. Play Protect may warn about an unknown developer: **More details → Install anyway**.
4. Updates install over the old version (same signing key every build), keeping the
   page's saved settings.

Arm64 only (any phone from the last ~8 years). Android 7.0+ (API 24, ML Kit's minimum).

## How it works

- `tool/sync_web.dart` copies `../web/index.html` to `assets/www/index.html`, wrapped in
  `<!doctype html><html><head>…viewport…</head><body>` + page + `</body></html>` (the
  page is a fragment; its hosting service adds the same skeleton). CI always re-syncs, so
  the committed copy may be stale. After changing the web app, run it yourself:

  ```
  cd mobile
  flutter pub get
  dart run tool/sync_web.dart
  ```

- The app loads it with `loadHtmlString(html, baseUrl: "https://appassets.androidplatform.net/")`,
  which gives the page a normal https origin (localStorage, blob workers and Google Fonts
  work; offline it falls back to system fonts). That host is Android's reserved WebView
  asset domain, so nothing is fetched from it. Links open in the browser; Back exits.
- The page's `<input type=file>` also works (camera if `capture` is set, else the photo
  picker), as a fallback.

### Bridge contract

The app adds a JavaScript channel `PixelHost`. The page sends
`PixelHost.postMessage(JSON.stringify({id, method, args}))`; the app answers with
`window.__pixelHostReply(id, ok, result)` (`result` is JSON, or an error string when
`ok` is false).

| method | args | result |
|---|---|---|
| `pickImage` | `{source: "camera" \| "gallery"}` | `data:image/jpeg;base64,…`, long side ≤ 1600 px, EXIF rotation baked in. Transparent images come back as PNG; small upright JPEG/PNG/WebP pass through untouched. Cancel → `ok=false`, `"cancelled"` |
| `segment` | `{image: dataURL}` | `data:image/png;base64,…` mask, same size as the image: grey = alpha = foreground confidence (white/opaque = subject). ML Kit Subject Segmentation, foreground confidence mask |
| `save` | `{filename, base64, mime}` | Images (`image/png`, jpeg, gif, webp) → gallery album "Pixel Pipeline": `"Saved to Gallery"`. Anything else (`.zip` …) → Android share sheet (Drive, Files, Quick Share, mail): `"Shared"`, or `"Share cancelled"` if dismissed |
| `info` | `{}` | `{platform: "android", version: "1.0.N", segment: true}` |

Replies longer than 512 K characters (a photo's data URL is 0.5–3 MB) are delivered in
several `runJavaScript` calls that fill `window.__pixelHostChunks[id]`, then one call
that joins, parses, deletes the buffer and calls `__pixelHostReply` as usual. Android's
`evaluateJavascript` has no documented size limit; the chunking is a cheap safeguard.
Page → app messages (e.g. a multi-MB `save`) go in one `postMessage`; Chromium's IPC
limit is far above anything the page sends.

### Code

- `lib/bridge/bridge.dart`: message parsing, routing, reply scripts + chunking
- `lib/bridge/handlers.dart`: the four methods, against interfaces in `lib/host/host_services.dart`
- `lib/host/device_services.dart`: plugin implementations (image_picker, ML Kit, gal, share_plus, package_info_plus)
- `lib/image_ops.dart`: pure-Dart image work (EXIF rotation, resize, mask PNG), run in a background isolate
- `lib/web_shell.dart`, `lib/app.dart`, `lib/main.dart`: WebView, splash, wiring
- `tool/make_icon.py`: draws the 16×16 pixel-art launcher icon (`../.venv/bin/python tool/make_icon.py`)

## Develop

```
cd mobile
flutter pub get
dart run tool/sync_web.dart
flutter analyze
flutter test
flutter run                  # phone with USB debugging
flutter build apk --release --split-per-abi --target-platform android-arm64
```

Debug builds allow `chrome://inspect` on the PC to debug the page inside the app.

## Signing

Debug and release builds are signed with `android/app/sideload.keystore` (committed,
password `android`). It's not a secret and not a Play Store key: it only exists so every
build, local or CI, has the same signature and installs over the previous one. If you
ever publish to the Play Store, make a real upload key and keep it out of git.

## Caveats

- **Cut-out needs Google Play services.** The ML Kit model is a Play services module,
  downloaded on first use (the manifest also requests it at install). While it
  downloads, `segment` retries for 20 s, then fails with "still downloading … try again
  in a minute", and the page falls back to its own background removal.
- No permissions are requested for photos (Android photo picker / camera app) or for
  saving on Android 10+. Android 7–10 asks for storage permission on the first gallery save.
- If Android kills the app while the camera is open (low memory), the photo is lost and
  the page reloads; take it again.
