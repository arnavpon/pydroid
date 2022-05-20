import 'dart:async';
import 'dart:developer';

import 'package:flutter/services.dart';

class Pydroid {
  // The class users of the plugin invoke

  static const MethodChannel _channel = MethodChannel('pydroid');

  static Future<String?> get platformVersion async {
    final String? version = await _channel.invokeMethod('getPlatformVersion');
    log("[flutter] Getting platform version: $version");
    return version;
  }

  // Practice Example

  static Future<int?> runTest() async {
    log("[flutter] pydroid.dart - runTest called...");
    final int? result = await _channel.invokeMethod('test');
    return result;
  }

  static Future<String?> execute() async {
    log("[flutter] pydroid.dart - called execute...");
    final String? result = await _channel.invokeMethod('execute');
    return result;
  }

  static Future<String?> executeInBackground(
      int iterations, int modelSize) async {
    log("[flutter] pydroid.dart - called executeInBackground...");
    final String? result =
        await _channel.invokeMethod('executeInBackground', <String, int>{
      "iterations": iterations,
      "modelSize": modelSize,
    });
    return result;
  }

  // static Future<int?> onKeyUp(int key) async {
  //   final int? numNotesOn = await _channel.invokeMethod('onKeyUp', [key]);
  //   // can pass parameters to the invokeMethod as a list
  //   return numNotesOn;
  // }

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
