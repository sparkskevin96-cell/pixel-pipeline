import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:pixel_pipeline/app.dart';
import 'package:pixel_pipeline/bridge/handlers.dart';
import 'package:pixel_pipeline/host/host_services.dart';
import 'package:pixel_pipeline/web_shell.dart';

import 'fakes.dart';

/// Stands in for the WebView.
class FakePage implements WebPageController {
  String? html;
  String? baseUrl;
  PageCallbacks? callbacks;
  final List<String> scripts = [];

  @override
  Widget buildView(BuildContext context) => const ColoredBox(key: Key('page'), color: Colors.white);

  @override
  Future<void> load({
    required String html,
    required String baseUrl,
    required PageCallbacks callbacks,
  }) async {
    this.html = html;
    this.baseUrl = baseUrl;
    this.callbacks = callbacks;
  }

  @override
  Future<void> runJavaScript(String script) async => scripts.add(script);
}

void main() {
  testWidgets('app builds, loads the wrapped page, splashes until loaded, answers the bridge', (
    tester,
  ) async {
    final page = FakePage();
    await tester.pumpWidget(
      PixelPipelineApp(
        createPage: (_) => page,
        loadHtml: () async => '<title>Pixel Pipeline</title><p>fragment</p>',
        handlers: buildHandlers(
          HostServices(
            photos: FakePhotoSource(),
            segmenter: FakeSegmenter(),
            saver: FakeSaver(),
            appInfo: FakeAppInfo(),
          ),
        ),
      ),
    );
    await tester.pump();

    expect(find.byKey(const Key('page')), findsOneWidget);
    expect(find.text('Pixel Pipeline'), findsOneWidget); // splash
    expect(page.baseUrl, pageBaseUrl);
    expect(page.html, startsWith('<!doctype html><html><head><meta charset="utf-8">'));
    expect(page.html, endsWith('<p>fragment</p></body></html>'));

    // The page calls info() at boot.
    page.callbacks!.onMessage('{"id":"c1","method":"info","args":{}}');
    await tester.pump();
    expect(page.scripts, [
      'window.__pixelHostReply&&window.__pixelHostReply("c1",true,'
          '{"platform":"android","version":"1.2.3","segment":true});',
    ]);

    page.callbacks!.onProgress(60);
    await tester.pump();
    expect(tester.widget<LinearProgressIndicator>(find.byType(LinearProgressIndicator)).value, 0.6);

    page.callbacks!.onLoaded();
    await tester.pumpAndSettle();
    final splash = tester.widget<AnimatedOpacity>(find.byType(AnimatedOpacity));
    expect(splash.opacity, 0);
  });

  testWidgets('a page that fails to load shows an error instead of spinning forever', (
    tester,
  ) async {
    await tester.pumpWidget(
      PixelPipelineApp(
        createPage: (_) => FakePage(),
        loadHtml: () async => throw Exception('asset missing'),
        handlers: const {},
      ),
    );
    await tester.pump();
    expect(find.textContaining('asset missing'), findsOneWidget);
  });
}
