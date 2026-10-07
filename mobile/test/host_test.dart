import 'dart:async';

import 'package:flutter/services.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:pixel_pipeline/host/device_services.dart';
import 'package:pixel_pipeline/host/host_services.dart';
import 'package:pixel_pipeline/web_shell.dart';

void main() {
  group('retryWhile', () {
    late DateTime now;
    late List<Duration> slept;

    setUp(() {
      now = DateTime(2026);
      slept = [];
    });

    Future<void> sleep(Duration d) async {
      slept.add(d);
      now = now.add(d);
    }

    test('retries transient errors until the action succeeds', () async {
      var calls = 0;
      final result = await retryWhile(
        () async {
          if (++calls < 3) {
            throw PlatformException(code: 'x', message: 'Waiting for the module to be downloaded');
          }
          return 'mask';
        },
        retryIf: isModelDownloadPending,
        timeout: const Duration(seconds: 45),
        sleep: sleep,
        now: () => now,
      );
      expect(result, 'mask');
      expect(calls, 3);
      expect(slept, hasLength(2));
    });

    test('gives up with a TimeoutException after the timeout', () async {
      var calls = 0;
      await expectLater(
        retryWhile(
          () async {
            calls++;
            throw PlatformException(code: 'x', message: 'optional module not downloaded');
          },
          retryIf: isModelDownloadPending,
          timeout: const Duration(seconds: 10),
          sleep: sleep,
          now: () => now,
        ),
        throwsA(isA<TimeoutException>()),
      );
      expect(calls, 5); // t = 0, 2, 4, 6, 8; the next wait would pass the deadline
    });

    test('other errors are rethrown immediately', () async {
      var calls = 0;
      await expectLater(
        retryWhile<int>(
          () async {
            calls++;
            throw PlatformException(code: 'x', message: 'Bad image');
          },
          retryIf: isModelDownloadPending,
          timeout: const Duration(seconds: 10),
          sleep: sleep,
          now: () => now,
        ),
        throwsA(isA<PlatformException>()),
      );
      expect(calls, 1);
    });
  });

  test('isModelDownloadPending recognises ML Kit\'s module-download error', () {
    expect(
      isModelDownloadPending(
        PlatformException(
          code: 'Subject segmentation failure!',
          message:
              'Waiting for the subject segmentation optional module to be downloaded. Please wait.',
        ),
      ),
      isTrue,
    );
    expect(
      isModelDownloadPending(PlatformException(code: 'x', message: 'Failed to decode bitmap')),
      isFalse,
    );
  });

  test('acceptsImages decides when the file chooser can use the photo picker', () {
    expect(acceptsImages(const ['image/*']), isTrue);
    expect(acceptsImages(const []), isTrue);
    expect(acceptsImages(const ['']), isTrue);
    expect(acceptsImages(const ['.png,.jpg']), isTrue);
    expect(acceptsImages(const ['application/zip']), isFalse);
    expect(acceptsImages(const ['.gpl', '.hex']), isFalse);
  });
}
