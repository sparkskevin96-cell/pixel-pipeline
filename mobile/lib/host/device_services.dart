/// Real implementations of the host services, backed by Flutter plugins.
library;

import 'dart:async';
import 'dart:io';

import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart';
import 'package:gal/gal.dart';
import 'package:google_mlkit_subject_segmentation/google_mlkit_subject_segmentation.dart';
import 'package:image_picker/image_picker.dart';
import 'package:package_info_plus/package_info_plus.dart';
import 'package:path_provider/path_provider.dart';
import 'package:share_plus/share_plus.dart';

import '../bridge/bridge.dart';
import '../bridge/handlers.dart' show galleryAlbum;
import '../image_ops.dart' show maxPhotoSide;
import 'host_services.dart';

HostServices deviceHostServices() => HostServices(
  photos: ImagePickerPhotoSource(),
  segmenter: MlKitSubjectSegmenter(),
  saver: DeviceFileSaver(),
  appInfo: PackageAppInfo(),
);

/// image_picker: Android's photo picker / camera app. Downscales natively (long side
/// [maxPhotoSide]); `preparePhoto` then bakes in the EXIF rotation, which
/// image_picker only copies over.
class ImagePickerPhotoSource implements PhotoSource {
  final ImagePicker _picker = ImagePicker();

  @override
  Future<Uint8List?> pickImage(PhotoSourceKind source) async {
    final file = await _pick(source);
    if (file == null) return null;
    final bytes = await file.readAsBytes();
    unawaited(_deleteQuietly(file.path)); // the picker's scaled copy in our cache
    return bytes;
  }

  @override
  Future<String?> pickImageFile(PhotoSourceKind source) async => (await _pick(source))?.path;

  Future<XFile?> _pick(PhotoSourceKind source) async {
    try {
      return await _picker.pickImage(
        source: source == PhotoSourceKind.camera ? ImageSource.camera : ImageSource.gallery,
        maxWidth: maxPhotoSide.toDouble(),
        maxHeight: maxPhotoSide.toDouble(),
        imageQuality: 92,
        preferredCameraDevice: CameraDevice.rear,
      );
    } on PlatformException catch (e) {
      throw BridgeException(switch (e.code) {
        'camera_access_denied' =>
          'Camera permission denied. Allow it in Android Settings > Apps > Pixel Pipeline.',
        'photo_access_denied' =>
          'Photo access denied. Allow it in Android Settings > Apps > Pixel Pipeline.',
        'no_available_camera' => 'No camera app found on this phone.',
        _ => e.message ?? e.code,
      });
    }
  }
}

/// Google ML Kit subject segmentation (on-device, via Google Play services). The model
/// is an optional Play services module: the manifest asks for it at install, and the
/// first call also triggers the download, so "still downloading" errors are retried
/// for up to [downloadWait].
class MlKitSubjectSegmenter implements SubjectSegmenterService {
  MlKitSubjectSegmenter({this.downloadWait = const Duration(seconds: 20)});

  final Duration downloadWait;
  SubjectSegmenter? _segmenter;

  @override
  bool get isSupported => !kIsWeb && Platform.isAndroid;

  @override
  Future<List<double>> foregroundConfidence(
    Uint8List encoded,
    String mimeType,
    int width,
    int height,
  ) async {
    final dir = await getTemporaryDirectory();
    final ext = mimeType == 'image/png' ? 'png' : 'jpg';
    final file = File('${dir.path}/segment_${DateTime.now().microsecondsSinceEpoch}.$ext');
    await file.writeAsBytes(encoded, flush: true);
    try {
      final segmenter = _segmenter ??= SubjectSegmenter(
        options: SubjectSegmenterOptions(
          enableForegroundConfidenceMask: true,
          enableForegroundBitmap: false,
          enableMultipleSubjects: SubjectResultOptions(
            enableConfidenceMask: false,
            enableSubjectBitmap: false,
          ),
        ),
      );
      final result = await retryWhile(
        () => segmenter.processImage(InputImage.fromFilePath(file.path)),
        retryIf: isModelDownloadPending,
        timeout: downloadWait,
        onRetry: (e, n) => debugPrint('segment: model not ready (attempt $n): $e'),
      );
      final mask = result.foregroundConfidenceMask;
      if (mask == null || mask.isEmpty) {
        throw const BridgeException('The cut-out model returned no mask');
      }
      return mask;
    } on TimeoutException {
      throw const BridgeException(
        'The cut-out model is still downloading through Google Play services. '
        'Stay online and try again in a minute.',
      );
    } on PlatformException catch (e) {
      throw BridgeException('On-device cut-out failed: ${e.message ?? e.code}');
    } finally {
      unawaited(_deleteQuietly(file.path));
    }
  }
}

/// ML Kit's "optional module not downloaded yet" error (MlKitException.UNAVAILABLE:
/// "Waiting for the subject segmentation optional module to be downloaded").
bool isModelDownloadPending(Object error) {
  final text = (error is PlatformException ? '${error.code} ${error.message}' : '$error')
      .toLowerCase();
  return text.contains('download') ||
      text.contains('waiting for') ||
      text.contains('optional module');
}

/// Images -> gallery album via gal (MediaStore; no permission on Android 10+).
/// Other files -> Android share sheet via share_plus (Drive, Files, Quick Share, mail...).
class DeviceFileSaver implements FileSaver {
  @override
  Future<void> saveImageToGallery(Uint8List bytes, String filename) async {
    if (!await Gal.hasAccess(toAlbum: true) && !await Gal.requestAccess(toAlbum: true)) {
      throw const BridgeException('No permission to save to the gallery');
    }
    final dot = filename.lastIndexOf('.');
    final name = dot > 0 ? filename.substring(0, dot) : filename;
    try {
      await Gal.putImageBytes(bytes, album: galleryAlbum, name: name);
    } on GalException catch (e) {
      throw BridgeException(switch (e.type) {
        GalExceptionType.accessDenied => 'No permission to save to the gallery',
        GalExceptionType.notEnoughSpace => 'Not enough storage space',
        GalExceptionType.notSupportedFormat => 'The gallery does not accept this image format',
        GalExceptionType.unexpected =>
          'Could not save to the gallery: ${e.platformException.message}',
      });
    }
  }

  @override
  Future<bool> shareFile(Uint8List bytes, String filename, String mimeType) async {
    final outbox = Directory('${(await getTemporaryDirectory()).path}/outbox');
    // Files from earlier shares: the receiving app has had its chance to read them.
    if (outbox.existsSync()) outbox.deleteSync(recursive: true);
    outbox.createSync(recursive: true);
    final file = File('${outbox.path}/$filename');
    await file.writeAsBytes(bytes, flush: true);
    final result = await SharePlus.instance.share(
      ShareParams(
        files: [XFile(file.path, mimeType: mimeType, name: filename)],
        title: filename,
      ),
    );
    return result.status != ShareResultStatus.dismissed;
  }
}

class PackageAppInfo implements AppInfo {
  @override
  Future<String> version() async => (await PackageInfo.fromPlatform()).version;
}

Future<void> _deleteQuietly(String path) async {
  try {
    await File(path).delete();
  } catch (_) {}
}
