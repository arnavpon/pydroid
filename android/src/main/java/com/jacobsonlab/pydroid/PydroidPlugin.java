package com.jacobsonlab.pydroid;

import android.app.Activity;

import androidx.annotation.NonNull;

import io.flutter.embedding.engine.plugins.FlutterPlugin;
import io.flutter.embedding.engine.plugins.activity.ActivityPluginBinding;
import io.flutter.plugin.common.BinaryMessenger;
import io.flutter.plugin.common.MethodCall;
import io.flutter.plugin.common.MethodChannel;
import io.flutter.plugin.common.MethodChannel.MethodCallHandler;
import io.flutter.plugin.common.MethodChannel.Result;
import io.flutter.plugin.common.PluginRegistry.Registrar;

import io.flutter.embedding.engine.plugins.activity.ActivityAware;


import java.util.ArrayList;  // ***


/** PydroidPlugin */
public class PydroidPlugin implements FlutterPlugin, MethodCallHandler, ActivityAware {
  /// The MethodChannel that handles communication between Flutter and native Android
  ///
  /// This local reference serves to register the plugin with the Flutter Engine and unregister it
  /// when the Flutter Engine is detached from the Activity
  private MethodChannel channel;
  private PythonHandler pyHandler;
  private static final String channelName = "pydroid";
  private android.content.Context context;
  private Activity activity;

  private Synth synth;

  // Mark: - Activity Aware Implementation

  @Override
  public void onAttachedToActivity(@NonNull ActivityPluginBinding binding) {
    activity = binding.getActivity();
  }

  @Override
  public void onReattachedToActivityForConfigChanges(@NonNull ActivityPluginBinding binding) {
    activity = binding.getActivity();
  }

  @Override
  public void onDetachedFromActivity() {
    activity = null;
  }

  @Override
  public void onDetachedFromActivityForConfigChanges() {
    activity = null;
  }

  // MARK: - Plugin Methods

  // @Override
  // public void onAttachedToEngine(@NonNull FlutterPluginBinding flutterPluginBinding) {
  //   channel = new MethodChannel(flutterPluginBinding.getBinaryMessenger(), "pydroid");
  //   channel.setMethodCallHandler(this);
  // }

  @Override
  public void onAttachedToEngine(@NonNull FlutterPluginBinding flutterPluginBinding) {
    channel = new MethodChannel(flutterPluginBinding.getBinaryMessenger(), channelName);
    channel.setMethodCallHandler(this);
    context = flutterPluginBinding.getApplicationContext();
    pyHandler = new PythonHandler(context);  // init python handler

    // plugin.synth = new Synth();
    // plugin.synth.start();
  }

  @Override
  public void onMethodCall(@NonNull MethodCall call, @NonNull Result result) {
    if (call.method.equals("getPlatformVersion")) {
      result.success("Android " + android.os.Build.VERSION.RELEASE);
      pyHandler.test(); // ***

    } else if (call.method.equals("onKeyDown")) {
      System.out.println("[droid] onKeyDown...");
      try {
        ArrayList arguments = (ArrayList) call.arguments;
        int numKeysDown = synth.keyDown((Integer) arguments.get(0));
        result.success(numKeysDown);
      } catch (Exception ex) {
        result.error("1", ex.getMessage(), ex.getStackTrace());
      }
    } else if (call.method.equals("onKeyUp")) {
      System.out.println("[droid] onKeyUp...");
      try {
        ArrayList arguments = (ArrayList) call.arguments;
        int numKeysDown = synth.keyUp((Integer) arguments.get(0));
        result.success(numKeysDown);
      } catch (Exception ex) {
        result.error("1", ex.getMessage(), ex.getStackTrace());
      }
    } else {
      result.notImplemented();
    }
  }

  @Override
  public void onDetachedFromEngine(@NonNull FlutterPluginBinding binding) {
    channel.setMethodCallHandler(null);
  }
}
