/// Interfaces for the native features the bridge uses. The real implementations
/// (plugins) are in `device_services.dart`; tests use fakes.
library;

import 'dart:async';
import 'dart:typed_data';

enum PhotoSourceKind { camera, gallery }

/// Takes or picks a photo.
abstract interface class PhotoSource {
  /// The photo's bytes (already downscaled by the picker), or null if the user cancelled.
  Future<Uint8List?> pickImage(PhotoSourceKind source);

  /// Like [pickImage] but returns the file path (for the page's `<input type=file>`).
  Future<String?> pickImageFile(PhotoSourceKind source);
}

/// On-device subject segmentation.
abstract interface class SubjectSegmenterService {
  /// Whether this device can segment at all (Android with Google Play services).
  bool get isSupported;

  /// Foreground confidence (0..1) per pixel, row-major, `width * height` values, for
  /// an encoded upright JPEG/PNG of that size. Throws `BridgeException` with a
  /// readable message when segmentation isn't available.
  Future<List<double>> foregroundConfidence(
    Uint8List encoded,
    String mimeType,
    int width,
    int height,
  );
}

/// Puts files where the user can find them.
abstract interface class FileSaver {
  /// Saves an image into the gallery (album "Pixel Pipeline").
  Future<void> saveImageToGallery(Uint8List bytes, String filename);

  /// Offers any file through the system share sheet. False if the user dismissed it.
  Future<bool> shareFile(Uint8List bytes, String filename, String mimeType);
}

/// App metadata.
abstract interface class AppInfo {
  /// Version name, e.g. "1.0.42".
  Future<String> version();
}

/// Everything the bridge handlers need.
class HostServices {
  const HostServices({
    required this.photos,
    required this.segmenter,
    required this.saver,
    required this.appInfo,
  });

  final PhotoSource photos;
  final SubjectSegmenterService segmenter;
  final FileSaver saver;
  final AppInfo appInfo;
}

/// Retries [action] while [retryIf] says the error is transient (e.g. an ML model is
/// still downloading), waiting [interval] between attempts. Gives up with a
/// [TimeoutException] once [timeout] has passed; other errors are rethrown at once.
Future<T> retryWhile<T>(
  Future<T> Function() action, {
  required bool Function(Object error) retryIf,
  required Duration timeout,
  Duration interval = const Duration(seconds: 2),
  Future<void> Function(Duration delay)? sleep,
  DateTime Function()? now,
  void Function(Object error, int attempt)? onRetry,
}) async {
  final clock = now ?? DateTime.now;
  final wait = sleep ?? (d) => Future<void>.delayed(d);
  final deadline = clock().add(timeout);
  for (var attempt = 1; ; attempt++) {
    try {
      return await action();
    } catch (e) {
      if (!retryIf(e)) rethrow;
      if (!clock().add(interval).isBefore(deadline)) {
        throw TimeoutException('Still not ready after ${timeout.inSeconds} s: $e', timeout);
      }
      onRetry?.call(e, attempt);
      await wait(interval);
    }
  }
}
