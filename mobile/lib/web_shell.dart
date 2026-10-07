/// The WebView that shows the page, behind a small interface so widget tests can
/// swap in a fake (the real WebView needs the Android platform).
library;

import 'dart:async';

import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:url_launcher/url_launcher.dart';
import 'package:webview_flutter/webview_flutter.dart';
import 'package:webview_flutter_android/webview_flutter_android.dart';

import 'bridge/bridge.dart';
import 'host/host_services.dart';

/// Origin the page runs under. `loadHtmlString` with this base URL gives the page a
/// real https origin (localStorage, blob: workers, CORS fetches and Google Fonts behave
/// as on the hosted site). The host is Android's reserved WebView asset domain, so
/// nothing is ever fetched from it.
const String pageBaseUrl = 'https://appassets.androidplatform.net/';

/// Callbacks from the page.
class PageCallbacks {
  const PageCallbacks({required this.onMessage, required this.onProgress, required this.onLoaded});

  /// A `PixelHost.postMessage` string.
  final void Function(String message) onMessage;

  /// Load progress, 0..100.
  final void Function(int percent) onProgress;

  /// The page finished loading.
  final void Function() onLoaded;
}

abstract interface class WebPageController {
  /// The widget that shows the page.
  Widget buildView(BuildContext context);

  /// Sets up the bridge channel and loads [html] as if served from [baseUrl].
  Future<void> load({
    required String html,
    required String baseUrl,
    required PageCallbacks callbacks,
  });

  /// Runs a script in the page.
  Future<void> runJavaScript(String script);
}

/// webview_flutter implementation.
class WebViewPageController implements WebPageController {
  WebViewPageController({required this.photos, required this.backgroundColor});

  /// Used for the page's `<input type=file>` fallback.
  final PhotoSource photos;
  final Color backgroundColor;
  final WebViewController _controller = WebViewController();

  @override
  Widget buildView(BuildContext context) => WebViewWidget(controller: _controller);

  @override
  Future<void> runJavaScript(String script) => _controller.runJavaScript(script);

  @override
  Future<void> load({
    required String html,
    required String baseUrl,
    required PageCallbacks callbacks,
  }) async {
    final c = _controller;
    await c.setJavaScriptMode(JavaScriptMode.unrestricted);
    await c.setBackgroundColor(backgroundColor);
    await c.addJavaScriptChannel(
      PixelBridge.channelName,
      onMessageReceived: (m) => callbacks.onMessage(m.message),
    );
    await c.setNavigationDelegate(
      NavigationDelegate(
        onProgress: callbacks.onProgress,
        onPageFinished: (_) => callbacks.onLoaded(),
        onNavigationRequest: (request) => _navigation(request, baseUrl),
        onWebResourceError: (e) =>
            debugPrint('page: ${e.errorCode} ${e.description} ${e.url ?? ''}'),
      ),
    );
    final platform = c.platform;
    if (platform is AndroidWebViewController) {
      await AndroidWebViewController.enableDebugging(kDebugMode); // chrome://inspect
      await platform.setOnShowFileSelector(_showFileSelector);
      // The Flutter layout already keeps the page clear of the system bars and the
      // keyboard (SafeArea + Scaffold); stop the WebView applying those insets again.
      await platform.setInsetsForWebContentToIgnore(const [
        AndroidWebViewInsets.systemBars,
        AndroidWebViewInsets.displayCutout,
        AndroidWebViewInsets.ime,
      ]);
      await platform.setOnConsoleMessage((m) {
        if (kDebugMode) debugPrint('console.${m.level.name}: ${m.message}');
      });
    }
    await c.loadHtmlString(html, baseUrl: baseUrl);
  }

  /// The page is the whole app: anything that would navigate away (links) opens in
  /// the browser instead.
  Future<NavigationDecision> _navigation(NavigationRequest request, String baseUrl) async {
    final url = request.url;
    if (!request.isMainFrame ||
        url.startsWith(baseUrl) ||
        url.startsWith('data:') ||
        url.startsWith('blob:') ||
        url == 'about:blank') {
      return NavigationDecision.navigate;
    }
    final uri = Uri.tryParse(url);
    if (uri != null) {
      unawaited(launchUrl(uri, mode: LaunchMode.externalApplication).catchError((_) => false));
    }
    return NavigationDecision.prevent;
  }

  /// `<input type=file accept="image/*" [capture]>`: camera when `capture` is set,
  /// else the photo picker. Other file types aren't offered (the page doesn't use them).
  Future<List<String>> _showFileSelector(FileSelectorParams params) async {
    if (params.mode == FileSelectorMode.save || !acceptsImages(params.acceptTypes)) return const [];
    try {
      final path = await photos.pickImageFile(
        params.isCaptureEnabled ? PhotoSourceKind.camera : PhotoSourceKind.gallery,
      );
      return path == null ? const [] : [Uri.file(path).toString()];
    } catch (e) {
      debugPrint('file selector: $e');
      return const [];
    }
  }
}

/// Whether an `<input accept>` list can be satisfied by an image picker.
bool acceptsImages(List<String> acceptTypes) {
  final types = acceptTypes
      .expand((t) => t.split(','))
      .map((t) => t.trim().toLowerCase())
      .where((t) => t.isNotEmpty);
  if (types.isEmpty) return true;
  const imageExtensions = {'.png', '.jpg', '.jpeg', '.webp', '.gif', '.heic', '.heif', '.bmp'};
  return types.any((t) => t.startsWith('image/') || imageExtensions.contains(t));
}
