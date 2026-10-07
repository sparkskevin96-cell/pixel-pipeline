import 'dart:async';
import 'dart:convert';
import 'dart:io';

import 'package:flutter/services.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:pixel_pipeline/bridge/bridge.dart';

/// Collects the scripts the bridge runs.
class ScriptLog {
  final List<String> scripts = [];
  Future<void> run(String script) async => scripts.add(script);
}

String request(String id, String method, [Map<String, Object?>? args]) =>
    jsonEncode({'id': id, 'method': method, 'args': ?args});

/// Reassembles a reply from the scripts with plain Dart (no JS engine): returns
/// (id, ok, result).
(String, bool, Object?) decodeReply(List<String> scripts) {
  final call = RegExp(
    r'window\.__pixelHostReply\(("(?:[^"\\]|\\.)*"),(true|false),(.*)\);\s*$',
    dotAll: true,
  );
  if (scripts.length == 1) {
    final m = call.firstMatch(scripts.single)!;
    return (jsonDecode(m[1]!) as String, m[2] == 'true', jsonDecode(m[3]!));
  }
  final push = RegExp(r'\.push\((.*)\);$', dotAll: true);
  final json = scripts
      .sublist(1, scripts.length - 1)
      .map((s) => jsonDecode(push.firstMatch(s)![1]!) as String)
      .join();
  final last = RegExp(r'__pixelHostReply\(("(?:[^"\\]|\\.)*"),(true|false),JSON\.parse')
      .firstMatch(scripts.last)!;
  return (jsonDecode(last[1]!) as String, last[2] == 'true', jsonDecode(json));
}

void main() {
  group('BridgeRequest.parse', () {
    test('reads id, method and args', () {
      final r = BridgeRequest.parse(request('c1', 'pickImage', {'source': 'camera'}));
      expect(r.id, 'c1');
      expect(r.method, 'pickImage');
      expect(r.args, {'source': 'camera'});
    });

    test('missing args is an empty map', () {
      expect(BridgeRequest.parse('{"id":"a","method":"info"}').args, isEmpty);
    });

    test('rejects malformed messages without an id', () {
      for (final raw in [
        'not json',
        '[1,2]',
        '{"method":"info"}',
        '{"id":7,"method":"info"}',
        '{"id":"","method":"x"}',
      ]) {
        expect(
          () => BridgeRequest.parse(raw),
          throwsA(isA<BadBridgeRequest>().having((e) => e.id, 'id', isNull)),
          reason: raw,
        );
      }
    });

    test('keeps the id when only method/args are wrong', () {
      expect(
        () => BridgeRequest.parse('{"id":"c9","method":3}'),
        throwsA(isA<BadBridgeRequest>().having((e) => e.id, 'id', 'c9')),
      );
      expect(
        () => BridgeRequest.parse('{"id":"c9","method":"save","args":"x"}'),
        throwsA(isA<BadBridgeRequest>().having((e) => e.id, 'id', 'c9')),
      );
    });
  });

  group('PixelBridge.handleMessage', () {
    late ScriptLog log;
    late List<Map<String, Object?>> seenArgs;
    late PixelBridge bridge;

    setUp(() {
      log = ScriptLog();
      seenArgs = [];
      bridge = PixelBridge(
        runScript: log.run,
        handlers: {
          'echo': (args) async {
            seenArgs.add(args);
            return args['value'];
          },
          'cancel': (args) async => throw const BridgeException('cancelled'),
          'platform': (args) async => throw PlatformException(code: 'E', message: 'native failed'),
          'boom': (args) async => throw StateError('kaput'),
          'plainException': (args) async => throw Exception('plain'),
          'notJson': (args) async => Object(),
        },
      );
    });

    test('routes to the handler and replies ok with its result', () async {
      await bridge.handleMessage(request('c1', 'echo', {'value': 'hi'}));
      expect(seenArgs.single, {'value': 'hi'});
      expect(log.scripts, ['window.__pixelHostReply&&window.__pixelHostReply("c1",true,"hi");']);
    });

    test('JSON results (objects, numbers, null) are inlined as JSON', () async {
      await bridge.handleMessage(
        request('c2', 'echo', {
          'value': {
            'a': 1,
            'b': [true, null],
            'c': 'x"y',
          },
        }),
      );
      final (id, ok, result) = decodeReply(log.scripts);
      expect((id, ok), ('c2', true));
      expect(result, {
        'a': 1,
        'b': [true, null],
        'c': 'x"y',
      });
      log.scripts.clear();
      await bridge.handleMessage(request('c3', 'echo'));
      expect(decodeReply(log.scripts), ('c3', true, null));
    });

    test('unknown method -> ok=false with a message', () async {
      await bridge.handleMessage(request('c4', 'nope'));
      expect(decodeReply(log.scripts), ('c4', false, 'Unknown method "nope"'));
    });

    test('BridgeException message is passed through ("cancelled")', () async {
      await bridge.handleMessage(request('c5', 'cancel'));
      expect(decodeReply(log.scripts), ('c5', false, 'cancelled'));
    });

    test('PlatformException / other errors become readable strings', () async {
      await bridge.handleMessage(request('c6', 'platform'));
      await bridge.handleMessage(request('c7', 'boom'));
      await bridge.handleMessage(request('c8', 'plainException'));
      expect(decodeReply([log.scripts[0]]), ('c6', false, 'native failed'));
      expect(decodeReply([log.scripts[1]]).$3, contains('kaput'));
      expect(decodeReply([log.scripts[2]]), ('c8', false, 'plain'));
    });

    test('a result that is not JSON-encodable -> ok=false', () async {
      await bridge.handleMessage(request('c9', 'notJson'));
      expect(decodeReply(log.scripts), ('c9', false, 'Internal error: result is not JSON'));
    });

    test('garbage without an id is dropped; with an id it gets an error reply', () async {
      await bridge.handleMessage('}{');
      expect(log.scripts, isEmpty);
      await bridge.handleMessage('{"id":"c10","args":{}}');
      final (id, ok, result) = decodeReply(log.scripts);
      expect((id, ok), ('c10', false));
      expect(result, startsWith('Bad request'));
    });

    test('a failing runScript (page gone) does not throw', () async {
      final b = PixelBridge(
        handlers: {'x': (_) async => 1},
        runScript: (_) async => throw StateError('no page'),
      );
      await expectLater(b.handleMessage(request('c11', 'x')), completes);
    });

    test('concurrent calls each get their own reply, in completion order', () async {
      final slow = Completer<String>();
      final b = PixelBridge(
        runScript: log.run,
        handlers: {'slow': (_) => slow.future, 'fast': (_) async => 'fast'},
      );
      final first = b.handleMessage(request('a', 'slow'));
      await b.handleMessage(request('b', 'fast'));
      slow.complete('slow');
      await first;
      expect(decodeReply([log.scripts[0]]), ('b', true, 'fast'));
      expect(decodeReply([log.scripts[1]]), ('a', true, 'slow'));
    });

    test('ids are escaped safely', () async {
      await bridge.handleMessage(request('we"ird\\id</script>', 'echo', {'value': 1}));
      expect(decodeReply(log.scripts), ('we"ird\\id</script>', true, 1));
    });
  });

  group('large replies', () {
    final big =
        'data:image/png;base64,${base64Encode(List.generate(3 * 1024 * 1024, (i) => i * 31 % 251))}';

    test('are split into chunks no longer than chunkSize', () {
      final scripts = PixelBridge.replyScripts('c1', true, big);
      expect(scripts.length, greaterThan(3));
      for (final s in scripts) {
        expect(s.length, lessThanOrEqualTo(PixelBridge.defaultChunkSize));
      }
      expect(scripts.first, contains('__pixelHostChunks'));
      expect(scripts.last, contains('JSON.parse'));
      expect(decodeReply(scripts), ('c1', true, big));
    });

    test('small replies stay one call', () {
      expect(PixelBridge.replyScripts('c1', true, 'data:image/png;base64,AAAA'), hasLength(1));
    });

    test('chunking never splits a surrogate pair', () {
      final text = '\u{1F436}' * 40; // dog emoji, 2 UTF-16 units each
      for (var size = 2; size < 9; size++) {
        final parts = splitForJs(text, size);
        expect(parts.join(), text);
        for (final p in parts) {
          expect(p.length, lessThanOrEqualTo(size));
          final last = p.codeUnitAt(p.length - 1);
          expect(last >= 0xD800 && last <= 0xDBFF, isFalse, reason: 'size $size');
        }
      }
    });

    final node = _findNode();
    test('the scripts deliver the exact result when run by a real JS engine (node)', () async {
      final tricky = 'quote " backslash \\ newline \n tab \t emoji \u{1F436} sep    </script> é';
      final cases = <(String, bool, Object?)>[
        ('c1', true, big),
        ('c2', true, tricky * 2000), // forces chunking with escapes and surrogates
        ('c3', false, 'cancelled'),
        ('c4', true, {'platform': 'android', 'version': '1.0.0', 'segment': true}),
      ];
      for (final (id, ok, result) in cases) {
        final scripts = PixelBridge.replyScripts(id, ok, result, chunkSize: 64 * 1024);
        final program = StringBuffer()
          ..writeln('const window = globalThis; let got = null;')
          ..writeln('window.__pixelHostReply = (id, ok, result) => { got = {id, ok, result}; };');
        for (final s in scripts) {
          program.writeln(s);
        }
        program.writeln(
          'if (Object.keys(window.__pixelHostChunks || {}).length) throw new Error("buffer left behind");',
        );
        program.writeln('process.stdout.write(JSON.stringify(got));');
        final dir = await Directory.systemTemp.createTemp('bridge_js');
        try {
          final file = File('${dir.path}/reply.js')..writeAsStringSync(program.toString());
          final run = await Process.run(node!, [file.path], stdoutEncoding: utf8);
          expect(run.exitCode, 0, reason: run.stderr.toString());
          final got = jsonDecode(run.stdout as String) as Map<String, Object?>;
          expect(got['id'], id);
          expect(got['ok'], ok);
          expect(got['result'], result);
        } finally {
          dir.deleteSync(recursive: true);
        }
      }
    }, skip: node == null ? 'node not installed' : false);
  });

  test('describeError', () {
    expect(describeError(const BridgeException('x')), 'x');
    expect(describeError(PlatformException(code: 'C')), 'C');
    expect(describeError(TimeoutException('slow')), 'slow');
    expect(describeError(const FormatException('bad')), 'bad');
    expect(describeError(Exception('e')), 'e');
  });
}

String? _findNode() {
  try {
    final r = Process.runSync('node', ['--version']);
    return r.exitCode == 0 ? 'node' : null;
  } catch (_) {
    return null;
  }
}
