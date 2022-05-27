import 'dart:async';
import 'dart:developer';
import 'dart:ui';
import 'dart:convert';

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';

class Pydroid {
  // The class users of the plugin invoke

  static const MethodChannel _channel = MethodChannel('pydroid');
  static const String KEY_SCRIPT = "key_python_script";
  static const String KEY_OUTPUT_VALUE =
      "key_output_value"; // matches key in plugin
  static const String KEY_OUTPUT_ERROR =
      "key_output_error"; // matches key in plugin

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

  static Future<Map<String, dynamic>> executeInBackground(
      String script, Map<String, dynamic> args) async {
    /// Calls the specified plugin platform in the background w/ a given arg list
    /// Input:
    /// - script: string | name of python script to call, OMIT the .py!
    /// - args: Map<string, object> | dictionary of arguments needed to run script
    /// Returns: Map<String, dynamic> | keys are either "value" or "error", value is dynamic

    log("[flutter] pydroid.dart - called executeInBackground...");
    args[KEY_SCRIPT] = script; // add script name to argument list
    Map<String, dynamic> returnValue = {};
    try {
      // only successful results are returned through this channel
      final String successfulResult = await _channel.invokeMethod(
          "executeInBackground", args); // result is JSON
      dynamic decodedResult = jsonDecode(successfulResult);
      returnValue[KEY_OUTPUT_VALUE] = decodedResult;
      log("[dart - executeInBackground] Successfully JSONdecoded result: ${decodedResult.toString()} | type: ${decodedResult.runtimeType.toString()}");
    } catch (e) {
      log("[dart - executeInBackground] Error: ${e.toString()}}");
      returnValue[KEY_OUTPUT_ERROR] = e.toString();
    }
    return returnValue;
  }

  static Future<String?> executeLinRegInBackground(
      int iterations, int modelSize) async {
    log("[flutter] pydroid.dart - executing LR script...");
    final result = await executeInBackground('lin_reg', {
      "iterations": iterations,
      "model_size": modelSize,
    });
    if (result[KEY_OUTPUT_ERROR] == null) {
      final value = result[KEY_OUTPUT_VALUE];
      return double.parse("$value").toStringAsFixed(2);
    } else {
      return "...error...";
    }
  }

  static Future<Rect> getFaceForImage(String imagePath) async {
    log("[flutter] pydroid.dart - getting face for image...");
    final result = await executeInBackground(
        'FaceDetection_MT1', {"image_path": imagePath});
    if (result[KEY_OUTPUT_ERROR] == null) {
      log("[dart] Converting object to rect...");
      // convert successful result to expected format (bounding box)
      Map<String, dynamic> value = result[KEY_OUTPUT_VALUE];
      final topLeft = Offset(value["x1"] as double, value["y1"] as double);
      final bottomRight = Offset(value["x2"] as double, value["y2"] as double);
      final rect = Rect.fromPoints(topLeft, bottomRight);
      log("Final rect: ${rect.toString()}");
      return rect;
    } else {
      log("[dart] error returned");
      return Rect.zero;
    }
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
