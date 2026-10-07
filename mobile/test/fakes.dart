import 'dart:typed_data';

import 'package:image/image.dart' as img;
import 'package:pixel_pipeline/host/host_services.dart';

class FakePhotoSource implements PhotoSource {
  Uint8List? next;
  final List<PhotoSourceKind> calls = [];

  @override
  Future<Uint8List?> pickImage(PhotoSourceKind source) async {
    calls.add(source);
    return next;
  }

  @override
  Future<String?> pickImageFile(PhotoSourceKind source) async => null;
}

class FakeSegmenter implements SubjectSegmenterService {
  FakeSegmenter({this.isSupported = true});

  @override
  final bool isSupported;

  /// Returns the mask for a width x height image; default: left half subject.
  List<double> Function(int width, int height) mask = (w, h) =>
      List.generate(w * h, (i) => (i % w) < w / 2 ? 1.0 : 0.0);
  Object? error;
  final List<(String, int, int)> calls = [];

  @override
  Future<List<double>> foregroundConfidence(
    Uint8List encoded,
    String mimeType,
    int width,
    int height,
  ) async {
    calls.add((mimeType, width, height));
    if (error != null) throw error!;
    return mask(width, height);
  }
}

class FakeSaver implements FileSaver {
  final List<(Uint8List, String)> gallery = [];
  final List<(Uint8List, String, String)> shared = [];
  bool dismissShare = false;

  @override
  Future<void> saveImageToGallery(Uint8List bytes, String filename) async =>
      gallery.add((bytes, filename));

  @override
  Future<bool> shareFile(Uint8List bytes, String filename, String mimeType) async {
    shared.add((bytes, filename, mimeType));
    return !dismissShare;
  }
}

class FakeAppInfo implements AppInfo {
  @override
  Future<String> version() async => '1.2.3';
}

/// A photo-like test image: [marker] colour in a 4x4 block at the top-left corner of the
/// stored pixels, grey elsewhere.
img.Image testImage(int width, int height, {img.Color? marker, int channels = 3}) {
  final im = img.Image(width: width, height: height, numChannels: channels);
  img.fill(
    im,
    color: channels == 4 ? img.ColorRgba8(128, 128, 128, 255) : img.ColorRgb8(128, 128, 128),
  );
  img.fillRect(im, x1: 0, y1: 0, x2: 3, y2: 3, color: marker ?? img.ColorRgb8(255, 0, 0));
  return im;
}

/// JPEG bytes with an EXIF orientation tag (1 = upright, 6 = rotate 90 deg clockwise).
Uint8List jpegWithOrientation(img.Image image, int orientation) {
  final jpg = img.encodeJpg(image, quality: 95);
  final exif = img.ExifData()..imageIfd.orientation = orientation;
  return img.injectJpgExif(jpg, exif) ?? (throw StateError('could not inject EXIF'));
}
