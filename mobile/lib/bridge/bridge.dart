/// The JavaScript bridge between the web page and the app.
///
/// Contract (the page implements the JS side):
/// * The app registers a JavaScript channel named [PixelBridge.channelName]
///   (`PixelHost`). The page detects the app with
///   `typeof window.PixelHost?.postMessage === "function"`.
/// * JS -> Dart: `PixelHost.postMessage(JSON.stringify({id, method, args}))`, `id` a string.
/// * Dart -> JS: `window.__pixelHostReply(id, ok, result)`. `result` is a JSON value, or
///   an error message string when `ok` is false.
///
/// Replies longer than [PixelBridge.chunkSize] characters (multi-MB data URLs) are sent
/// as several `runJavaScript` calls that append to a temporary buffer
/// (`window.__pixelHostChunks[id]`), followed by one call that parses it, deletes the
/// buffer and calls `__pixelHostReply` exactly as a single call would.
library;

import 'dart:async';
import 'dart:convert';

import 'package:flutter/services.dart' show PlatformException;

/// Handles one method call. Returns a JSON-encodable result; throws
/// [BridgeException] (or anything else) to reply `ok = false`.
typedef BridgeHandler = Future<Object?> Function(Map<String, Object?> args);

/// Runs a script in the page (WebViewController.runJavaScript).
typedef ScriptRunner = Future<void> Function(String script);

/// An error whose [message] is shown to the page as-is (`ok = false`).
class BridgeException implements Exception {
  const BridgeException(this.message);

  final String message;

  @override
  String toString() => message;
}

/// A request that could not be parsed. [id] is set when one could still be read,
/// so the page can be told what went wrong.
class BadBridgeRequest implements Exception {
  const BadBridgeRequest(this.message, {this.id});

  final String message;
  final String? id;

  @override
  String toString() => 'BadBridgeRequest($message, id: $id)';
}

/// One parsed `{id, method, args}` message.
class BridgeRequest {
  const BridgeRequest({required this.id, required this.method, required this.args});

  /// Parses the raw `postMessage` string. Throws [BadBridgeRequest].
  factory BridgeRequest.parse(String raw) {
    final Object? decoded;
    try {
      decoded = jsonDecode(raw);
    } on FormatException catch (e) {
      throw BadBridgeRequest('Message is not JSON: ${e.message}');
    }
    if (decoded is! Map) throw const BadBridgeRequest('Message is not a JSON object');
    final id = decoded['id'];
    if (id is! String || id.isEmpty) {
      throw const BadBridgeRequest('"id" must be a non-empty string');
    }
    final method = decoded['method'];
    if (method is! String || method.isEmpty) {
      throw BadBridgeRequest('"method" must be a non-empty string', id: id);
    }
    final args = decoded['args'];
    if (args != null && args is! Map) throw BadBridgeRequest('"args" must be an object', id: id);
    return BridgeRequest(
      id: id,
      method: method,
      args: args == null ? const {} : Map<String, Object?>.from(args as Map),
    );
  }

  final String id;
  final String method;
  final Map<String, Object?> args;
}

/// Routes page messages to [handlers] and sends the replies back.
class PixelBridge {
  PixelBridge({
    required Map<String, BridgeHandler> handlers,
    required this.runScript,
    this.chunkSize = defaultChunkSize,
    void Function(String message)? log,
  }) : _handlers = Map.unmodifiable(handlers),
       _log = log ?? _noLog;

  static void _noLog(String message) {}

  /// Name of the JavaScript channel (`window.PixelHost`).
  static const String channelName = 'PixelHost';

  /// The page's reply callback.
  static const String replyFunction = '__pixelHostReply';

  /// Temporary page global that collects chunks of a large reply.
  static const String chunkBuffer = '__pixelHostChunks';

  /// Largest script sent in one `runJavaScript` call. Android WebView's
  /// `evaluateJavascript` has no documented limit (it goes over Chromium IPC, not a
  /// URL), but splitting keeps each call modest and is cheap: a 3 MB data URL is
  /// 6 calls.
  static const int defaultChunkSize = 512 * 1024;

  final Map<String, BridgeHandler> _handlers;

  /// Runs reply scripts in the page.
  final ScriptRunner runScript;
  final void Function(String message) _log;

  /// Maximum characters per script; see [defaultChunkSize].
  final int chunkSize;

  /// Methods the bridge answers.
  Iterable<String> get methods => _handlers.keys;

  /// Handles one raw message from the page and sends the reply. Never throws.
  Future<void> handleMessage(String raw) async {
    final BridgeRequest request;
    try {
      request = BridgeRequest.parse(raw);
    } on BadBridgeRequest catch (e) {
      _log('bridge: bad request: ${e.message}');
      if (e.id != null) await _reply(e.id!, false, 'Bad request: ${e.message}');
      return;
    }

    final handler = _handlers[request.method];
    if (handler == null) {
      await _reply(request.id, false, 'Unknown method "${request.method}"');
      return;
    }

    Object? result;
    try {
      result = await handler(request.args);
    } catch (e, st) {
      if (e is! BridgeException) _log('bridge: ${request.method} failed: $e\n$st');
      await _reply(request.id, false, describeError(e));
      return;
    }
    await _reply(request.id, true, result);
  }

  Future<void> _reply(String id, bool ok, Object? result) async {
    List<String> scripts;
    try {
      scripts = replyScripts(id, ok, result, chunkSize: chunkSize);
    } on JsonUnsupportedObjectError catch (e) {
      _log('bridge: reply for $id is not JSON-encodable: $e');
      scripts = replyScripts(id, false, 'Internal error: result is not JSON', chunkSize: chunkSize);
    }
    try {
      for (final script in scripts) {
        await runScript(script);
      }
    } catch (e) {
      // The page may have reloaded or gone away; nothing else to do.
      _log('bridge: could not deliver reply $id: $e');
    }
  }

  /// The script(s) that deliver one reply, in order. Pure, for tests.
  ///
  /// Throws [JsonUnsupportedObjectError] when [result] is not JSON-encodable.
  static List<String> replyScripts(
    String id,
    bool ok,
    Object? result, {
    int chunkSize = defaultChunkSize,
  }) {
    final idJs = jsLiteral(id);
    final resultJson = _jsSafe(jsonEncode(result));
    final call = 'window.$replyFunction';
    final single = '$call&&$call($idJs,$ok,$resultJson);';
    if (single.length <= chunkSize) return [single];

    final buffer = 'window.$chunkBuffer';
    final scripts = <String>['($buffer=$buffer||{})[$idJs]=[];'];
    // Leave room for the push(...) wrapper and any escaping in the literal.
    final step = chunkSize > 2048 ? chunkSize - 1024 : chunkSize ~/ 2 + 1;
    for (final chunk in splitForJs(resultJson, step)) {
      scripts.add('$buffer[$idJs].push(${jsLiteral(chunk)});');
    }
    scripts.add(
      '(function(){var b=$buffer,p=b&&b[$idJs];if(!p)return;delete b[$idJs];'
      '$call&&$call($idJs,$ok,JSON.parse(p.join("")));})();',
    );
    return scripts;
  }
}

/// A readable message for an error thrown by a handler.
String describeError(Object error) {
  if (error is BridgeException) return error.message;
  if (error is PlatformException) return error.message ?? error.code;
  if (error is TimeoutException) return error.message ?? 'Timed out';
  if (error is FormatException) return error.message;
  final text = error.toString();
  return text.startsWith('Exception: ') ? text.substring(11) : text;
}

/// [value] as a JavaScript string literal (JSON-encoded, safe inside a script).
String jsLiteral(String value) => _jsSafe(jsonEncode(value));

/// JSON is a JavaScript expression except that U+2028/U+2029 were not allowed in
/// string literals before ES2019; escape them for old WebViews.
String _jsSafe(String json) => json.replaceAll(' ', r' ').replaceAll(' ', r' ');

/// Splits [text] into pieces of at most [size] UTF-16 code units without cutting a
/// surrogate pair in half (a lone surrogate would not survive the UTF-8 trip to the
/// WebView).
List<String> splitForJs(String text, int size) {
  if (size < 2) throw ArgumentError.value(size, 'size', 'must be at least 2');
  final pieces = <String>[];
  var start = 0;
  while (start < text.length) {
    var end = start + size;
    if (end >= text.length) {
      end = text.length;
    } else {
      final unit = text.codeUnitAt(end - 1);
      if (unit >= 0xD800 && unit <= 0xDBFF) end -= 1; // high surrogate: keep the pair together
    }
    pieces.add(text.substring(start, end));
    start = end;
  }
  return pieces;
}
