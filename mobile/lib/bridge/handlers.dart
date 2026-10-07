/// The bridge methods: `pickImage`, `segment`, `save`, `info`.
library;

import 'dart:isolate';
import 'dart:typed_data';

import '../host/host_services.dart';
import '../image_ops.dart';
import 'bridge.dart';
import 'data_url.dart';

/// Gallery album that `save` puts images in.
const String galleryAlbum = 'Pixel Pipeline';

/// Image types `save` puts in the gallery; everything else goes to the share sheet.
const Set<String> galleryMimeTypes = {'image/png', 'image/jpeg', 'image/gif', 'image/webp'};

/// Runs pure image work off the UI isolate. Tests may pass an inline runner.
typedef Compute = Future<R> Function<R>(R Function() work);

Future<R> _isolateRun<R>(R Function() work) => Isolate.run(work);

/// Bridge handlers backed by [services].
Map<String, BridgeHandler> buildHandlers(HostServices services, {Compute compute = _isolateRun}) {
  return {
    'pickImage': (args) => _pickImage(services, compute, args),
    'segment': (args) => _segment(services, compute, args),
    'save': (args) => _save(services, args),
    'info': (args) => _info(services),
  };
}

/// `{source: "camera" | "gallery"}` -> `data:image/jpeg;base64,...` (PNG if the picked
/// image has transparency). Cancel -> error "cancelled".
Future<Object?> _pickImage(HostServices s, Compute compute, Map<String, Object?> args) async {
  final source = switch (args['source']) {
    'camera' => PhotoSourceKind.camera,
    'gallery' => PhotoSourceKind.gallery,
    _ => throw const BridgeException('source must be "camera" or "gallery"'),
  };
  final bytes = await s.photos.pickImage(source);
  if (bytes == null) throw const BridgeException('cancelled');
  final EncodedImage photo;
  try {
    photo = await compute(() => preparePhoto(bytes));
  } on FormatException catch (e) {
    throw BridgeException(e.message);
  }
  return toDataUrl(photo.mimeType, photo.bytes);
}

/// `{image: dataURL}` -> `data:image/png;base64,...` mask, same size as the image.
Future<Object?> _segment(HostServices s, Compute compute, Map<String, Object?> args) async {
  if (!s.segmenter.isSupported) {
    throw const BridgeException('On-device cut-out is not available on this device');
  }
  final image = args['image'];
  if (image is! String) throw const BridgeException('image must be a data: URL');
  final Uint8List bytes;
  try {
    bytes = DataUrl.parse(image).bytes;
  } on FormatException catch (e) {
    throw BridgeException('image must be a base64 data: URL (${e.message})');
  }
  final EncodedImage input;
  try {
    input = await compute(() => prepareForSegmentation(bytes));
  } on FormatException catch (e) {
    throw BridgeException(e.message);
  }
  final confidence = await s.segmenter.foregroundConfidence(
    input.bytes,
    input.mimeType,
    input.width,
    input.height,
  );
  if (confidence.length != input.width * input.height) {
    throw BridgeException(
      'Cut-out mask has ${confidence.length} values for a '
      '${input.width}x${input.height} image',
    );
  }
  final (w, h) = (input.width, input.height);
  final png = await compute(() => encodeMaskPng(confidence, w, h));
  return toDataUrl('image/png', png);
}

/// `{filename, base64, mime}` -> "Saved to Gallery" (images) / "Shared" (other files).
Future<Object?> _save(HostServices s, Map<String, Object?> args) async {
  final data = args['base64'];
  if (data is! String || data.isEmpty) {
    throw const BridgeException('base64 must be a non-empty string');
  }
  final mimeArg = args['mime'];
  final filename = sanitizeFilename(args['filename'], mimeArg is String ? mimeArg : null);
  final mime = (mimeArg is String && mimeArg.contains('/'))
      ? mimeArg.split(';').first.trim().toLowerCase()
      : guessMimeType(filename);
  final Uint8List bytes;
  try {
    bytes = data.startsWith('data:') ? DataUrl.parse(data).bytes : decodeBase64(data);
  } on FormatException {
    throw const BridgeException('base64 is not valid base64');
  }
  if (bytes.isEmpty) throw const BridgeException('Nothing to save (empty file)');

  if (galleryMimeTypes.contains(mime)) {
    await s.saver.saveImageToGallery(bytes, filename);
    return 'Saved to Gallery';
  }
  final shared = await s.saver.shareFile(bytes, filename, mime);
  return shared ? 'Shared' : 'Share cancelled';
}

Future<Object?> _info(HostServices s) async => {
  'platform': 'android',
  'version': await s.appInfo.version(),
  'segment': s.segmenter.isSupported,
};

/// A safe file name: no folders or characters Android storage rejects, at most 120
/// characters, an extension guessed from [mimeType] when missing.
String sanitizeFilename(Object? name, [String? mimeType]) {
  var base = name is String ? name : '';
  base = base.split(RegExp(r'[/\\]')).last;
  base = base.replaceAll(RegExp(r'[\x00-\x1f\x7f:*?"<>|]'), '_').trim();
  base = base.replaceFirst(RegExp(r'^\.+'), '');
  if (base.isEmpty) base = 'pixel-pipeline';
  if (!base.contains('.')) {
    final ext = _extensionFor[mimeType?.toLowerCase()];
    if (ext != null) base = '$base.$ext';
  }
  if (base.length > 120) {
    final dot = base.lastIndexOf('.');
    final ext = dot > 0 && base.length - dot <= 12 ? base.substring(dot) : '';
    base = base.substring(0, 120 - ext.length) + ext;
  }
  return base;
}

/// MIME type from a file name's extension.
String guessMimeType(String filename) {
  final dot = filename.lastIndexOf('.');
  final ext = dot < 0 ? '' : filename.substring(dot + 1).toLowerCase();
  return switch (ext) {
    'png' => 'image/png',
    'jpg' || 'jpeg' => 'image/jpeg',
    'gif' => 'image/gif',
    'webp' => 'image/webp',
    'zip' => 'application/zip',
    'json' || 'tmj' => 'application/json',
    'txt' || 'tres' || 'gpl' || 'hex' => 'text/plain',
    _ => 'application/octet-stream',
  };
}

const Map<String?, String> _extensionFor = {
  'image/png': 'png',
  'image/jpeg': 'jpg',
  'image/gif': 'gif',
  'image/webp': 'webp',
  'application/zip': 'zip',
  'application/json': 'json',
  'text/plain': 'txt',
};
