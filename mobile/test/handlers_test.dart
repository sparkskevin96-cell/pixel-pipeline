import 'dart:convert';
import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:image/image.dart' as img;
import 'package:pixel_pipeline/bridge/bridge.dart';
import 'package:pixel_pipeline/bridge/data_url.dart';
import 'package:pixel_pipeline/bridge/handlers.dart';
import 'package:pixel_pipeline/host/host_services.dart';

import 'fakes.dart';

Future<R> inline<R>(R Function() work) async => work();

void main() {
  late FakePhotoSource photos;
  late FakeSegmenter segmenter;
  late FakeSaver saver;
  late Map<String, BridgeHandler> handlers;

  void build({bool segmentSupported = true, Compute compute = inline}) {
    photos = FakePhotoSource();
    segmenter = FakeSegmenter(isSupported: segmentSupported);
    saver = FakeSaver();
    handlers = buildHandlers(
      HostServices(photos: photos, segmenter: segmenter, saver: saver, appInfo: FakeAppInfo()),
      compute: compute,
    );
  }

  setUp(build);

  Future<Object?> call(String method, Map<String, Object?> args) => handlers[method]!(args);

  Matcher bridgeError(Object message) =>
      throwsA(isA<BridgeException>().having((e) => e.message, 'message', message));

  test('handles exactly the contract methods', () {
    expect(handlers.keys.toSet(), {'pickImage', 'segment', 'save', 'info'});
  });

  group('info', () {
    test('reports platform, version and segmentation support', () async {
      expect(await call('info', {}), {'platform': 'android', 'version': '1.2.3', 'segment': true});
      build(segmentSupported: false);
      expect((await call('info', {}) as Map)['segment'], false);
    });
  });

  group('pickImage', () {
    test('camera / gallery map to the picker sources', () async {
      photos.next = jpegWithOrientation(testImage(40, 20), 1);
      await call('pickImage', {'source': 'camera'});
      await call('pickImage', {'source': 'gallery'});
      expect(photos.calls, [PhotoSourceKind.camera, PhotoSourceKind.gallery]);
    });

    test('an upright, small JPEG passes through untouched', () async {
      final jpg = jpegWithOrientation(testImage(40, 20), 1);
      photos.next = jpg;
      final url = await call('pickImage', {'source': 'gallery'}) as String;
      expect(url, startsWith('data:image/jpeg;base64,'));
      expect(DataUrl.parse(url).bytes, jpg);
    });

    test('EXIF rotation is baked into the pixels', () async {
      // Stored 40x20 with a red block top-left; orientation 6 = displayed rotated
      // 90 deg clockwise, so the page should get 20x40 with the block top-right.
      photos.next = jpegWithOrientation(testImage(40, 20), 6);
      final url = await call('pickImage', {'source': 'camera'}) as String;
      expect(url, startsWith('data:image/jpeg;base64,'));
      final bytes = DataUrl.parse(url).bytes;
      expect(img.decodeJpgExif(bytes)?.imageIfd.orientation ?? 1, 1);
      final out = img.decodeJpg(bytes)!;
      expect((out.width, out.height), (20, 40));
      final p = out.getPixel(18, 1);
      expect(p.r, greaterThan(200));
      expect(p.g, lessThan(60));
    });

    test('the long side is limited to 1600 px', () async {
      photos.next = jpegWithOrientation(testImage(2000, 1000), 1);
      final url = await call('pickImage', {'source': 'gallery'}) as String;
      final out = img.decodeJpg(DataUrl.parse(url).bytes)!;
      expect((out.width, out.height), (1600, 800));
    });

    test('a transparent PNG stays a PNG (keeps its alpha)', () async {
      final im = testImage(16, 16, channels: 4, marker: img.ColorRgba8(0, 0, 0, 0));
      final png = img.encodePng(im);
      photos.next = png;
      final url = await call('pickImage', {'source': 'gallery'}) as String;
      expect(url, startsWith('data:image/png;base64,'));
      expect(DataUrl.parse(url).bytes, png);
    });

    test('cancel -> "cancelled"', () async {
      photos.next = null;
      await expectLater(call('pickImage', {'source': 'gallery'}), bridgeError('cancelled'));
    });

    test('bad source / unreadable file are readable errors', () async {
      await expectLater(call('pickImage', {'source': 'scanner'}), bridgeError(contains('camera')));
      await expectLater(call('pickImage', {}), bridgeError(contains('gallery')));
      photos.next = Uint8List.fromList(utf8.encode('not an image'));
      await expectLater(
        call('pickImage', {'source': 'gallery'}),
        bridgeError(contains("isn't an image")),
      );
    });

    test('works through a real background isolate', () async {
      handlers = buildHandlers(
        // default compute = Isolate.run
        HostServices(photos: photos, segmenter: segmenter, saver: saver, appInfo: FakeAppInfo()),
      );
      photos.next = jpegWithOrientation(testImage(40, 20), 6);
      final url = await call('pickImage', {'source': 'camera'}) as String;
      expect(img.decodeJpg(DataUrl.parse(url).bytes)!.width, 20);
    });
  });

  group('segment', () {
    String jpegUrl(int w, int h, {int orientation = 1}) =>
        toDataUrl('image/jpeg', jpegWithOrientation(testImage(w, h), orientation));

    test('returns a grey+alpha PNG mask the same size as the input', () async {
      segmenter.mask = (w, h) => List.generate(w * h, (i) => (i % w) / (w - 1));
      final url = await call('segment', {'image': jpegUrl(30, 20)}) as String;
      expect(url, startsWith('data:image/png;base64,'));
      expect(segmenter.calls.single, ('image/jpeg', 30, 20));
      final mask = img.decodePng(DataUrl.parse(url).bytes)!;
      expect((mask.width, mask.height), (30, 20));
      // Left edge: background (black, transparent). Right edge: subject (white, opaque).
      final bg = mask.getPixel(0, 10), fg = mask.getPixel(29, 10), mid = mask.getPixel(15, 5);
      expect((bg.r, bg.a), (0, 0));
      expect((fg.r, fg.a), (255, 255));
      expect(mid.r, inInclusiveRange(125, 140));
      expect(mid.a, mid.r);
    });

    test('a JPEG with EXIF rotation is segmented upright (mask matches the page)', () async {
      final url = await call('segment', {'image': jpegUrl(30, 20, orientation: 6)}) as String;
      expect(segmenter.calls.single, ('image/png', 20, 30));
      final mask = img.decodePng(DataUrl.parse(url).bytes)!;
      expect((mask.width, mask.height), (20, 30));
    });

    test('a PNG input works too', () async {
      final png = toDataUrl('image/png', img.encodePng(testImage(8, 6)));
      final url = await call('segment', {'image': png}) as String;
      expect(img.decodePng(DataUrl.parse(url).bytes)!.width, 8);
    });

    test('a wrong-sized mask is an error, not a misaligned image', () async {
      segmenter.mask = (w, h) => List.filled(w * h - 1, 1.0);
      await expectLater(call('segment', {'image': jpegUrl(10, 10)}), bridgeError(contains('mask')));
    });

    test('segmenter errors pass through; unsupported devices say so', () async {
      segmenter.error = const BridgeException('The cut-out model is still downloading');
      await expectLater(
        call('segment', {'image': jpegUrl(10, 10)}),
        bridgeError(contains('downloading')),
      );
      build(segmentSupported: false);
      await expectLater(
        call('segment', {'image': jpegUrl(10, 10)}),
        bridgeError(contains('not available')),
      );
    });

    test('bad input is a readable error', () async {
      await expectLater(call('segment', {}), bridgeError(contains('data: URL')));
      await expectLater(
        call('segment', {'image': 'https://x/y.png'}),
        bridgeError(contains('data: URL')),
      );
      await expectLater(
        call('segment', {'image': toDataUrl('image/png', utf8.encode('nope'))}),
        bridgeError(contains('readable')),
      );
    });
  });

  group('save', () {
    final pngBytes = img.encodePng(testImage(4, 4));

    test('PNG -> gallery', () async {
      final r = await call('save', {
        'filename': 'rex.sprite.png',
        'base64': base64Encode(pngBytes),
        'mime': 'image/png',
      });
      expect(r, 'Saved to Gallery');
      expect(saver.gallery.single.$1, pngBytes);
      expect(saver.gallery.single.$2, 'rex.sprite.png');
      expect(saver.shared, isEmpty);
    });

    test('zip -> share sheet', () async {
      final zip = Uint8List.fromList([0x50, 0x4b, 3, 4, 1, 2, 3]);
      final r = await call('save', {
        'filename': 'rex.sprite.zip',
        'base64': base64Encode(zip),
        'mime': 'application/zip',
      });
      expect(r, 'Shared');
      final (bytes, name, mime) = saver.shared.single;
      expect(bytes, zip);
      expect((name, mime), ('rex.sprite.zip', 'application/zip'));
      expect(saver.gallery, isEmpty);
    });

    test('dismissed share sheet -> "Share cancelled"', () async {
      saver.dismissShare = true;
      final r = await call('save', {
        'filename': 'a.zip',
        'base64': 'AAEC',
        'mime': 'application/zip',
      });
      expect(r, 'Share cancelled');
    });

    test('mime is guessed from the name; data URLs and unpadded base64 are accepted', () async {
      await call('save', {'filename': 'a.png', 'base64': toDataUrl('image/png', pngBytes)});
      expect(saver.gallery.single.$1, pngBytes);
      await call('save', {'filename': 'b.json', 'base64': 'e30'}); // "{}" without padding
      expect(saver.shared.single.$3, 'application/json');
      expect(utf8.decode(saver.shared.single.$1), '{}');
    });

    test('file names are sanitised', () async {
      await call('save', {
        'filename': '../../etc/pa:ss?.zip',
        'base64': 'AAEC',
        'mime': 'application/zip',
      });
      expect(saver.shared.single.$2, 'pa_ss_.zip');
    });

    test('bad input is a readable error', () async {
      await expectLater(call('save', {'filename': 'a.png'}), bridgeError(contains('base64')));
      await expectLater(
        call('save', {'filename': 'a.png', 'base64': '!!!'}),
        bridgeError(contains('base64')),
      );
    });
  });

  group('helpers', () {
    test('sanitizeFilename', () {
      expect(sanitizeFilename('rex.sprite.png'), 'rex.sprite.png');
      expect(sanitizeFilename(null, 'image/png'), 'pixel-pipeline.png');
      expect(sanitizeFilename('  ', 'application/zip'), 'pixel-pipeline.zip');
      expect(sanitizeFilename(r'C:\Users\x\sprite', 'image/png'), 'sprite.png');
      expect(sanitizeFilename('.hidden.zip'), 'hidden.zip');
      final long = sanitizeFilename('${'a' * 300}.aseprite');
      expect(long.length, 120);
      expect(long, endsWith('.aseprite'));
    });

    test('guessMimeType', () {
      expect(guessMimeType('a.PNG'), 'image/png');
      expect(guessMimeType('a.jpeg'), 'image/jpeg');
      expect(guessMimeType('a.zip'), 'application/zip');
      expect(guessMimeType('a.tres'), 'text/plain');
      expect(guessMimeType('a.aseprite'), 'application/octet-stream');
      expect(guessMimeType('noext'), 'application/octet-stream');
    });

    test('DataUrl round trip and errors', () {
      final url = toDataUrl('image/png', [1, 2, 3]);
      expect(url, 'data:image/png;base64,AQID');
      final d = DataUrl.parse(url);
      expect(d.mimeType, 'image/png');
      expect(d.bytes, [1, 2, 3]);
      expect(DataUrl.parse('data:;base64,AQID').mimeType, 'text/plain');
      expect(() => DataUrl.parse('data:image/png,abc'), throwsFormatException);
      expect(() => DataUrl.parse('nope'), throwsFormatException);
    });
  });
}
