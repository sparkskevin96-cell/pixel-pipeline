import 'dart:async';

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';

import 'bridge/bridge.dart';
import 'html_wrap.dart';
import 'web_shell.dart';

/// The web app's background colours (`--bg` in web/index.html), so the shell, the
/// status bar strip and the launch screen match the page.
const Color lightBackground = Color(0xFFEEF1F6);
const Color darkBackground = Color(0xFF11151D);
const Color _accent = Color(0xFF3B5DC9);

/// Builds the page controller once the theme brightness is known.
typedef WebPageFactory = WebPageController Function(Brightness brightness);

class PixelPipelineApp extends StatelessWidget {
  const PixelPipelineApp({
    super.key,
    required this.createPage,
    required this.handlers,
    required this.loadHtml,
  });

  final WebPageFactory createPage;
  final Map<String, BridgeHandler> handlers;

  /// Returns the page (a fragment or a full document).
  final Future<String> Function() loadHtml;

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Pixel Pipeline',
      debugShowCheckedModeBanner: false,
      themeMode: ThemeMode.system,
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: _accent, surface: lightBackground),
        scaffoldBackgroundColor: lightBackground,
      ),
      darkTheme: ThemeData(
        colorScheme: ColorScheme.fromSeed(
          seedColor: _accent,
          brightness: Brightness.dark,
          surface: darkBackground,
        ),
        scaffoldBackgroundColor: darkBackground,
      ),
      home: ShellScreen(createPage: createPage, handlers: handlers, loadHtml: loadHtml),
    );
  }
}

/// Full-screen page with a short splash while it loads. Back exits the app (the page
/// has no navigation of its own).
class ShellScreen extends StatefulWidget {
  const ShellScreen({
    super.key,
    required this.createPage,
    required this.handlers,
    required this.loadHtml,
  });

  final WebPageFactory createPage;
  final Map<String, BridgeHandler> handlers;
  final Future<String> Function() loadHtml;

  @override
  State<ShellScreen> createState() => _ShellScreenState();
}

class _ShellScreenState extends State<ShellScreen> {
  WebPageController? _page;
  late PixelBridge _bridge;
  int _progress = 0;
  bool _loaded = false;
  String? _error;

  @override
  void didChangeDependencies() {
    super.didChangeDependencies();
    if (_page != null) return;
    final page = widget.createPage(Theme.of(context).brightness);
    _page = page;
    _bridge = PixelBridge(
      handlers: widget.handlers,
      runScript: page.runJavaScript,
      log: (m) => debugPrint(m),
    );
    unawaited(_load(page));
  }

  Future<void> _load(WebPageController page) async {
    try {
      final html = wrapFragment(await widget.loadHtml());
      await page.load(
        html: html,
        baseUrl: pageBaseUrl,
        callbacks: PageCallbacks(
          onMessage: (m) => unawaited(_bridge.handleMessage(m)),
          onProgress: (p) {
            if (mounted) setState(() => _progress = p);
          },
          onLoaded: () {
            if (mounted) setState(() => _loaded = true);
          },
        ),
      );
    } catch (e) {
      if (mounted) setState(() => _error = 'Could not load the app page: $e');
    }
  }

  @override
  Widget build(BuildContext context) {
    final dark = Theme.of(context).brightness == Brightness.dark;
    final background = dark ? darkBackground : lightBackground;
    return AnnotatedRegion<SystemUiOverlayStyle>(
      value: (dark ? SystemUiOverlayStyle.light : SystemUiOverlayStyle.dark).copyWith(
        statusBarColor: Colors.transparent,
        systemNavigationBarColor: background,
      ),
      child: Scaffold(
        backgroundColor: background,
        body: SafeArea(
          child: Stack(
            fit: StackFit.expand,
            children: [
              _page!.buildView(context),
              IgnorePointer(
                ignoring: _loaded,
                child: AnimatedOpacity(
                  opacity: _loaded ? 0 : 1,
                  duration: const Duration(milliseconds: 250),
                  child: _Splash(background: background, progress: _progress, error: _error),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}

class _Splash extends StatelessWidget {
  const _Splash({required this.background, required this.progress, this.error});

  final Color background;
  final int progress;
  final String? error;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    return ColoredBox(
      color: background,
      child: Stack(
        children: [
          if (error == null)
            LinearProgressIndicator(
              value: progress > 0 && progress < 100 ? progress / 100 : null,
              minHeight: 3,
            ),
          Center(
            child: Padding(
              padding: const EdgeInsets.all(24),
              child: Column(
                mainAxisSize: MainAxisSize.min,
                children: [
                  Image.asset(
                    'assets/icon/sprite.png',
                    width: 96,
                    height: 96,
                    filterQuality: FilterQuality.none,
                    errorBuilder: (context, error, stack) => const SizedBox(width: 96, height: 96),
                  ),
                  const SizedBox(height: 16),
                  Text('Pixel Pipeline', style: theme.textTheme.titleMedium),
                  if (error != null) ...[
                    const SizedBox(height: 12),
                    Text(
                      error!,
                      textAlign: TextAlign.center,
                      style: theme.textTheme.bodyMedium?.copyWith(color: theme.colorScheme.error),
                    ),
                  ],
                ],
              ),
            ),
          ),
        ],
      ),
    );
  }
}
