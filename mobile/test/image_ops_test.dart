import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:image/image.dart' as img;
import 'package:pixel_pipeline/image_ops.dart';

import 'fakes.dart';

void main() {
  group('probeImage', () {
    test('reads JPEG size and EXIF orientation (displayed size)', () {
      final p = probeImage(jpegWithOrientation(testImage(40, 20), 6))!;
      expect((p.format, p.width, p.height, p.orientation), (img.ImageFormat.jpg, 20, 40, 6));
      final q = probeImage(jpegWithOrientation(testImage(40, 20), 1))!;
      expect((q.width, q.height, q.isUpright), (40, 20, true));
    });

    test('PNG, and garbage', () {
      final p = probeImage(img.encodePng(testImage(7, 5)))!;
      expect((p.format, p.width, p.height, p.orientation), (img.ImageFormat.png, 7, 5, 1));
      expect(probeImage(Uint8List.fromList([1, 2, 3, 4])), isNull);
    });
  });

  group('preparePhoto', () {
    test('rotations 3, 6 and 8 put the stored top-left corner where it is displayed', () {
      // Expected displayed position of the stored top-left 4x4 block (in a 40x20 image).
      final expected = {3: (37, 17), 6: (18, 1), 8: (1, 37)};
      expected.forEach((orientation, xy) {
        final out = preparePhoto(jpegWithOrientation(testImage(40, 20), orientation));
        expect(out.mimeType, 'image/jpeg');
        final im = img.decodeJpg(out.bytes)!;
        final p = im.getPixel(xy.$1, xy.$2);
        expect(p.r, greaterThan(200), reason: 'orientation $orientation');
        expect(p.g, lessThan(60), reason: 'orientation $orientation');
        expect((out.width, out.height), (im.width, im.height));
      });
    });

    test('portrait images are limited by their height', () {
      final out = preparePhoto(img.encodePng(testImage(500, 2400)));
      expect((out.width, out.height), (333, 1600));
      expect(out.mimeType, 'image/jpeg'); // opaque PNG that had to be re-encoded
    });

    test('maxSide is respected for any format', () {
      final out = preparePhoto(img.encodeBmp(testImage(30, 10)), maxSide: 15);
      expect((out.width, out.height), (15, 5));
    });
  });

  group('encodeMaskPng', () {
    test('grey = alpha = confidence x 255, clamped, NaN -> 0', () {
      final png = encodeMaskPng([0, 0.5, 1, 1.5, -1, double.nan], 3, 2);
      final mask = img.decodePng(png)!;
      expect((mask.width, mask.height, mask.numChannels), (3, 2, 2));
      final values = [for (final p in mask) (p.r.toInt(), p.a.toInt())];
      expect(values, [(0, 0), (128, 128), (255, 255), (255, 255), (0, 0), (0, 0)]);
    });

    test('rejects a length that does not match the size', () {
      expect(() => encodeMaskPng([1, 1, 1], 2, 2), throwsArgumentError);
    });

    test('is fast and small enough for a 1600x1200 mask', () {
      final w = 1600, h = 1200;
      final conf = Float32List(w * h);
      for (var y = 300; y < 900; y++) {
        for (var x = 400; x < 1200; x++) {
          conf[y * w + x] = 1;
        }
      }
      final sw = Stopwatch()..start();
      final png = encodeMaskPng(conf, w, h);
      expect(sw.elapsed, lessThan(const Duration(seconds: 5)));
      expect(png.length, lessThan(200 * 1024));
    });
  });
}
