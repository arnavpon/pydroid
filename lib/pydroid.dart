import 'dart:async';
import 'dart:developer';

import 'package:flutter/services.dart';

class Pydroid {
  // The class users of the plugin invoke

  static const MethodChannel _channel = MethodChannel('pydroid');

  static Future<String?> get platformVersion async {
    final String? version = await _channel.invokeMethod('getPlatformVersion');
    log("[Platform Version] Version: $version");
    return version;
  }

  // Practice Example

  static Future<int?> onKeyDown(int key) async {
    final int? numNotesOn = await _channel.invokeMethod('onKeyDown', [key]);
    // can pass parameters to the invokeMethod as a list
    return numNotesOn;
  }

  static Future<int?> onKeyUp(int key) async {
    final int? numNotesOn = await _channel.invokeMethod('onKeyUp', [key]);
    return numNotesOn;
  }

  /// A class that handles native python computations on native background threads
  /// returns dynamic data w/ type metadata?
  /// installation of desired python packages using pip at start

  static Future<Object?> runInlineScript(String code) async {
    // runs python inline
    final Object? result = await _channel.invokeMethod('executeInlineCode');
    return result;
  }

  static Future<Object?> runScriptWithArguments() async {
    // use invokeMapMethod to input args
  }
}
