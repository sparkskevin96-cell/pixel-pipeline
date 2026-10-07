import 'dart:io';

import 'package:flutter_test/flutter_test.dart';
import 'package:pixel_pipeline/html_wrap.dart';

void main() {
  test('wraps a fragment in the same skeleton as the hosting service', () {
    final html = wrapFragment('<title>Pixel Pipeline</title><style></style><p>hi</p>');
    expect(
      html,
      '<!doctype html><html><head><meta charset="utf-8">'
      '<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">'
      '</head><body><title>Pixel Pipeline</title><style></style><p>hi</p></body></html>',
    );
  });

  test('is idempotent and leaves full documents alone', () {
    final once = wrapFragment('<title>x</title>');
    expect(wrapFragment(once), once);
    expect(wrapFragment('  <!DOCTYPE html><html></html>'), '  <!DOCTYPE html><html></html>');
    expect(wrapFragment('<HTML><body></body></HTML>'), '<HTML><body></body></HTML>');
  });

  test('drops a byte-order mark', () {
    expect(wrapFragment('﻿<title>x</title>'), startsWith('<!doctype html>'));
    expect(isFullDocument('﻿<!doctype html>'), isTrue);
  });

  test('the bundled asset is a wrapped page with the bridge client', () {
    final asset = File('assets/www/index.html');
    expect(asset.existsSync(), isTrue, reason: 'run: dart run tool/sync_web.dart');
    final html = asset.readAsStringSync();
    expect(isFullDocument(html), isTrue);
    expect(html, contains('__pixelHostReply'));
  });
}
