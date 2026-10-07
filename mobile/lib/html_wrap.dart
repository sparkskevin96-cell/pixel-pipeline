/// The web app (`web/index.html`) is a page *fragment*: it starts with `<title>` and
/// `<style>` and has no `<html>`/`<head>`/`<body>`, because the hosting service it is
/// published to adds that skeleton. The app adds the same skeleton before loading it.
///
/// Pure Dart (no Flutter imports) so `tool/sync_web.dart` can use it too.
library;

/// Document start put in front of a fragment.
const String documentHead =
    '<!doctype html><html><head><meta charset="utf-8">'
    '<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">'
    '</head><body>';

/// Document end put after a fragment.
const String documentTail = '</body></html>';

/// Whether [html] is already a full document (starts with a doctype or `<html>`).
bool isFullDocument(String html) {
  final start = _stripBom(html).trimLeft();
  final probe = start.substring(0, start.length < 32 ? start.length : 32).toLowerCase();
  return probe.startsWith('<!doctype') || probe.startsWith('<html');
}

/// Wraps a fragment in [documentHead]/[documentTail]. Idempotent: a full document is
/// returned unchanged, so re-wrapping an already synced asset is harmless.
String wrapFragment(String html) {
  final body = _stripBom(html);
  if (isFullDocument(body)) return body;
  return '$documentHead$body$documentTail';
}

String _stripBom(String s) => s.startsWith('﻿') ? s.substring(1) : s;
