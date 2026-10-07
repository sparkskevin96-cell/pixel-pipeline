import 'package:flutter/material.dart';
import 'package:flutter/services.dart';

import 'app.dart';
import 'bridge/handlers.dart';
import 'host/device_services.dart';
import 'web_shell.dart';

void main() {
  WidgetsFlutterBinding.ensureInitialized();
  final services = deviceHostServices();
  runApp(
    PixelPipelineApp(
      handlers: buildHandlers(services),
      // The wrapped copy of ../web/index.html (`dart run tool/sync_web.dart`).
      loadHtml: () => rootBundle.loadString('assets/www/index.html', cache: false),
      createPage: (brightness) => WebViewPageController(
        photos: services.photos,
        backgroundColor: brightness == Brightness.dark ? darkBackground : lightBackground,
      ),
    ),
  );
}
