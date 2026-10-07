/// Pure-Dart image work for the bridge (no Flutter or platform code), so it is unit
/// tested and can run in a background isolate (`Isolate.run`).
library;

import 'dart:math' as math;
import 'dart:typed_data';

import 'package:image/image.dart' as img;

/// Long side of photos handed to the page.
const int maxPhotoSide = 1600;

/// JPEG quality for photos the app re-encodes.
const int photoJpegQuality = 90;

/// An encoded image and its size as displayed (EXIF orientation applied).
class EncodedImage {
  const EncodedImage(this.bytes, this.mimeType, this.width, this.height);

  final Uint8List bytes;
  final String mimeType;
  final int width;
  final int height;
}

/// What [probeImage] learns from the header, without decoding pixels.
class ImageProbe {
  const ImageProbe(this.format, this.width, this.height, this.orientation);

  final img.ImageFormat format;

  /// Size as displayed, i.e. swapped for EXIF orientations 5-8.
  final int width;
  final int height;

  /// EXIF orientation (1 = upright; JPEG only, 1 for other formats).
  final int orientation;

  bool get isUpright => orientation == 1;
}

/// Format, displayed size and orientation, or null when [bytes] isn't a readable image.
ImageProbe? probeImage(Uint8List bytes) {
  final img.Decoder? decoder;
  final img.DecodeInfo? info;
  try {
    // The `image` decoders can throw (RangeError etc.) on junk instead of returning null.
    decoder = img.findDecoderForData(bytes);
    info = decoder?.startDecode(bytes);
  } catch (_) {
    return null;
  }
  if (decoder == null || info == null || info.width <= 0 || info.height <= 0) return null;
  var orientation = 1;
  if (decoder.format == img.ImageFormat.jpg) {
    int? o;
    try {
      o = img.decodeJpgExif(bytes)?.imageIfd.orientation;
    } catch (_) {}
    if (o != null && o >= 1 && o <= 8) orientation = o;
  }
  final swap = orientation >= 5;
  return ImageProbe(
    decoder.format,
    swap ? info.height : info.width,
    swap ? info.width : info.height,
    orientation,
  );
}

/// A picked photo, ready for the page: EXIF orientation applied, long side at most
/// [maxSide]. Upright JPEG/PNG/WebP/GIF that are small enough pass through untouched
/// (PNG stays lossless, e.g. existing pixel art). Anything else is decoded and
/// re-encoded: JPEG, or PNG when it has transparency. Throws [FormatException].
EncodedImage preparePhoto(
  Uint8List bytes, {
  int maxSide = maxPhotoSide,
  int jpegQuality = photoJpegQuality,
}) {
  final probe = probeImage(bytes);
  if (probe == null) throw const FormatException("That file isn't an image the app can read");
  final passThrough =
      probe.isUpright &&
      math.max(probe.width, probe.height) <= maxSide &&
      _webMime.containsKey(probe.format);
  if (passThrough) return EncodedImage(bytes, _webMime[probe.format]!, probe.width, probe.height);

  var image = _decodeUpright(bytes);
  if (math.max(image.width, image.height) > maxSide) {
    final landscape = image.width >= image.height;
    image = img.copyResize(
      image,
      width: landscape ? maxSide : null,
      height: landscape ? null : maxSide,
      interpolation: img.Interpolation.average,
    );
  }
  image.exif = img.ExifData(); // drop camera metadata (GPS etc.); orientation is baked in
  if (_hasTransparency(image)) {
    final rgba = image.convert(format: img.Format.uint8, numChannels: 4);
    return EncodedImage(
      img.encodePng(rgba, singleFrame: true),
      'image/png',
      rgba.width,
      rgba.height,
    );
  }
  final rgb = image.convert(format: img.Format.uint8, numChannels: 3);
  return EncodedImage(
    img.encodeJpg(rgb, quality: jpegQuality),
    'image/jpeg',
    rgb.width,
    rgb.height,
  );
}

/// An image for the segmenter: upright JPEG/PNG pass through; anything else (or a
/// JPEG with an EXIF rotation) becomes a PNG with the rotation baked in, so the mask
/// lines up with what the page shows. Throws [FormatException].
EncodedImage prepareForSegmentation(Uint8List bytes) {
  final probe = probeImage(bytes);
  if (probe == null) throw const FormatException("The image to segment isn't a readable image");
  if (probe.isUpright &&
      (probe.format == img.ImageFormat.jpg || probe.format == img.ImageFormat.png)) {
    return EncodedImage(bytes, _webMime[probe.format]!, probe.width, probe.height);
  }
  final image = _decodeUpright(bytes)..exif = img.ExifData();
  final rgba = image.convert(format: img.Format.uint8, numChannels: 4);
  return EncodedImage(
    img.encodePng(rgba, singleFrame: true, level: 1),
    'image/png',
    rgba.width,
    rgba.height,
  );
}

/// A grey + alpha PNG from foreground confidences (0..1, row-major, [width] x
/// [height]): both grey and alpha = confidence x 255, so white/opaque = subject and
/// black/transparent = background.
Uint8List encodeMaskPng(List<double> confidence, int width, int height) {
  if (width <= 0 || height <= 0) throw ArgumentError('Mask size must be positive');
  if (confidence.length != width * height) {
    throw ArgumentError('Mask has ${confidence.length} values, expected ${width * height}');
  }
  final data = Uint8List(width * height * 2);
  for (var i = 0, o = 0; i < confidence.length; i++, o += 2) {
    final c = confidence[i];
    final v = c.isNaN ? 0 : (c <= 0 ? 0 : (c >= 1 ? 255 : (c * 255).round()));
    data[o] = v;
    data[o + 1] = v;
  }
  final mask = img.Image.fromBytes(
    width: width,
    height: height,
    bytes: data.buffer,
    numChannels: 2,
  );
  return img.encodePng(mask, singleFrame: true);
}

const Map<img.ImageFormat, String> _webMime = {
  img.ImageFormat.jpg: 'image/jpeg',
  img.ImageFormat.png: 'image/png',
  img.ImageFormat.webp: 'image/webp',
  img.ImageFormat.gif: 'image/gif',
};

img.Image _decodeUpright(Uint8List bytes) {
  img.Image? decoded;
  try {
    decoded = img.decodeImage(bytes);
  } catch (_) {}
  if (decoded == null) throw const FormatException("That file isn't an image the app can read");
  // The JPEG decoder already applies EXIF orientation; this covers other formats.
  final ifd = decoded.exif.imageIfd;
  return ifd.hasOrientation && ifd.orientation != 1 ? img.bakeOrientation(decoded) : decoded;
}

bool _hasTransparency(img.Image image) {
  if (!image.hasAlpha) return false;
  final opaque = image.maxChannelValue;
  for (final p in image) {
    if (p.a < opaque) return true;
  }
  return false;
}
