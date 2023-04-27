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

  // Examples

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

  static Future<Object?> runInlineScript(String code) async {
    // runs python inline
    final Object? result = await _channel.invokeMethod('executeInlineCode');
    return result;
  }

  static Future<Object?> runScriptWithArguments() async {
    // use invokeMapMethod to input args
  }

  // MARK: - Facial Detection Methods

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

  static Future<Rect> getFaceForImage(String imagePath) async {
    log("[flutter] pydroid.dart - getting face for image...");
    final result = await executeInBackground(
        'FaceDetection_MT1', {"image_path": imagePath});
    if (result[KEY_OUTPUT_ERROR] == null) {
      log("[dart] Converting object to rect...");
      // convert successful result to expected format (bounding box)
      Map<String, dynamic> value = result[KEY_OUTPUT_VALUE];

      double x1 = value["x1"] as double;
      double y1 = value["y1"] as double;
      double x2 = value["x2"] as double;
      double y2 = value["y2"] as double;

      double factor = 0.25;
      x1 = x1 + ((x2 - x1) * factor);
      y1 = y1 + ((y2 - y1) * factor);
      x2 = x2 - ((x2 - x1) * factor);
      y2 = y2 - ((y2 - y1) * (factor + 0.2));

      // final topLeft = Offset(value["x1"] as double, value["y1"] as double);
      // final topLeft = Offset(x1, value["y1"] as double);
      // final bottomRight = Offset(value["x2"] as double, value["y2"] as double);
      final topLeft = Offset(x1, y1);
      final bottomRight = Offset(x2, y2);
      final rect = Rect.fromPoints(topLeft, bottomRight);
      log("Final rect: ${rect.toString()}");
      return rect;
    } else {
      log("[dart] error returned");
      return Rect.zero;
    }
  }

  static Future<List<dynamic>> analyzeVideo(String videoPath) async {
    final result = await executeInBackground(
        'VideoFaceDetection', {"vid_path": videoPath});
    print('we got here');
    
    if (result[KEY_OUTPUT_ERROR] == null) {
      List<dynamic> value = result[KEY_OUTPUT_VALUE];
      log("[dart] we got the return");
      print(value);
      return value;
    } else {
      log("[dart] error returned");
      return [{}];
    }
  }

  static Future<Rect> analyzeStream(String path, String tracker_path) async {

    print('Executing...');
    final result = await executeInBackground(
      'AnalyzeStream', {"img_path": path, "tracker_path": tracker_path}
    );

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
}

