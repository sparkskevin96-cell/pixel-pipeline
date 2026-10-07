import 'dart:convert';
import 'dart:typed_data';

/// A decoded `data:<mime>;base64,<payload>` URL.
class DataUrl {
  const DataUrl(this.mimeType, this.bytes);

  /// Parses a base64 data URL. Throws [FormatException].
  factory DataUrl.parse(String url) {
    final s = url.trim();
    if (!s.startsWith('data:')) throw const FormatException('Not a data: URL');
    final comma = s.indexOf(',');
    if (comma < 0) throw const FormatException('Data URL has no payload');
    final meta = s.substring(5, comma).split(';');
    if (!meta.skip(1).any((p) => p.trim().toLowerCase() == 'base64')) {
      throw const FormatException('Data URL is not base64-encoded');
    }
    final mime = meta.first.trim().toLowerCase();
    return DataUrl(mime.isEmpty ? 'text/plain' : mime, decodeBase64(s.substring(comma + 1)));
  }

  final String mimeType;
  final Uint8List bytes;

  @override
  String toString() => toDataUrl(mimeType, bytes);
}

/// `data:<mime>;base64,...` for [bytes].
String toDataUrl(String mimeType, List<int> bytes) =>
    'data:$mimeType;base64,${base64Encode(bytes)}';

/// Decodes standard or URL-safe base64, ignoring whitespace and missing padding.
/// Throws [FormatException].
Uint8List decodeBase64(String text) {
  var s = text.replaceAll(RegExp(r'\s+'), '');
  if (s.contains('-') || s.contains('_')) s = s.replaceAll('-', '+').replaceAll('_', '/');
  final pad = s.length % 4;
  if (pad == 1) throw const FormatException('Invalid base64 length');
  if (pad != 0) s = s.padRight(s.length + 4 - pad, '=');
  return base64Decode(s);
}
