import 'dart:async';

import 'package:flutter/services.dart';

class Pydroid {
  static const MethodChannel _channel = MethodChannel('pydroid');

  static Future<String?> get platformVersion async {
    final String? version = await _channel.invokeMethod('getPlatformVersion');
    return version;
  }
}

/// A class that handles native python computations on native background threads
/// returns dynamic data w/ type metadata?
/// installation of desired python packages using pip at start
class PythonInterpreter {
  dynamic runInlineScript(String code) {
    // runs python inline
  }

  dynamic runScript(String script) {
    // runs single python script, how to handle imports & such?
  }
}
