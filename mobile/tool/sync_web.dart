// Copies ../web/index.html (a page fragment) into assets/www/index.html, wrapped in a
// full document, so the app can load it. CI runs this before every build; run it
// yourself after changing the web app:
//
//   cd mobile
//   flutter pub get            (once)
//   dart run tool/sync_web.dart
//
// Options: --src <path to fragment> --out <path to asset>.

import 'dart:io';

import 'package:pixel_pipeline/html_wrap.dart';

void main(List<String> args) {
  final mobileDir = _mobileDir();
  var src = File('${mobileDir.path}/../web/index.html');
  var out = File('${mobileDir.path}/assets/www/index.html');
  for (var i = 0; i < args.length; i++) {
    if (args[i] == '--src' && i + 1 < args.length) src = File(args[++i]);
    if (args[i] == '--out' && i + 1 < args.length) out = File(args[++i]);
  }

  if (!src.existsSync()) {
    stderr.writeln('sync_web: ${src.path} not found');
    exit(1);
  }
  final fragment = src.readAsStringSync();
  if (fragment.trim().isEmpty) {
    stderr.writeln('sync_web: ${src.path} is empty');
    exit(1);
  }
  final html = wrapFragment(fragment);
  out.parent.createSync(recursive: true);
  out.writeAsStringSync(html);
  stdout.writeln(
    'sync_web: ${src.path} -> ${out.path} '
    '(${html.length} chars${isFullDocument(fragment) ? ', already a full document' : ', wrapped'})',
  );
}

/// The `mobile/` folder: next to this script's `tool/` folder, else the working directory.
Directory _mobileDir() {
  final script = Platform.script;
  if (script.scheme == 'file' && script.path.endsWith('/tool/sync_web.dart')) {
    return File.fromUri(script).parent.parent;
  }
  return Directory.current;
}
