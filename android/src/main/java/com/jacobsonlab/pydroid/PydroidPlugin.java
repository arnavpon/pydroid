package com.jacobsonlab.pydroid;

import java.util.List;

import android.app.Activity;
import android.os.Looper;

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


/** PydroidPlugin */
public class PydroidPlugin implements FlutterPlugin, MethodCallHandler, ActivityAware {
  /// The MethodChannel that handles communication between Flutter and native Android
  /// This local reference serves to register the plugin with the Flutter Engine and unregister it when the Flutter Engine is detached from the Activity
  
  private MethodChannel channel;
  private PythonHandler pyHandler;
  private static final String channelName = "pydroid";
  private android.content.Context context;
  private Activity activity;

  // Mark: - Activity Aware Implementation

  @Override
  public void onAttachedToActivity(@NonNull ActivityPluginBinding binding) {
    System.out.println("[java] onAttachedToActivity");
    activity = binding.getActivity();
  }

  @Override
  public void onReattachedToActivityForConfigChanges(@NonNull ActivityPluginBinding binding) {
    System.out.println("[java] onReattachedToActivityForConfigChanges");
    activity = binding.getActivity();
  }

  @Override
  public void onDetachedFromActivity() {
    System.out.println("[java] onDetachedFromActivity");
    activity = null;
  }

  @Override
  public void onDetachedFromActivityForConfigChanges() {
    System.out.println("[java] onDetachedFromActivityForConfigChanges");
    activity = null;
  }

  // MARK: - Plugin Methods

  @Override
  public void onAttachedToEngine(@NonNull FlutterPluginBinding flutterPluginBinding) {
    // this function IS being called, we just don't see the output in the console!
    System.out.println("[onAttachedToEngine] setting up...");
    channel = new MethodChannel(flutterPluginBinding.getBinaryMessenger(), channelName);
    channel.setMethodCallHandler(this);
    context = flutterPluginBinding.getApplicationContext();
    pyHandler = new PythonHandler(context);  // init python handler
  }

  @Override
  public void onMethodCall(@NonNull MethodCall call, @NonNull Result result) {
    System.out.println("[onMethodCall] passing method call...");
    if (call.method.equals("getPlatformVersion")) {
      try {
        result.success("Android " + android.os.Build.VERSION.RELEASE);
      } catch (Exception ex) {
        result.error("1", ex.getMessage(), null);
      }

    } else if (call.method.equals("test")) {
      System.out.println("[java] running Python test...");
      try {
        if (pyHandler == null) {
          System.out.println("[java] python handler is null, initialize???");
          // no access to context from here
          // pyHandler = new PythonHandler(context);
        } else {
          System.out.println("[java] python handler is NOT null");
        }
        int res = pyHandler.test().length();
        result.success(res);
      } catch (Exception ex) {
        result.error("1", ex.getMessage(), null);
      }

    } else if (call.method.equals("execute")) {
      System.out.println("[droid] execute...");
      try {
        if (pyHandler == null) {
          System.out.println("[java] python handler is null, initialize???");
          // no access to context from here
          // pyHandler = new PythonHandler(context);
        } else {
          System.out.println("[java] python handler is NOT null");
        }
        result.success(pyHandler.executeScriptSync());
      } catch (Exception ex) {
        result.error("1", ex.getMessage(), null);
      }

    }  else if (call.method.equals("executeInBackground")) {
        System.out.println("[droid] executeInBackground...");
        new Thread(new Runnable() {
          public void run() {
            try {
              // ERROR - don't call result methods outside of main thread!
              // causes app to crash
              // how do we call back to main?
//              Looper.getMainLooper();
              String output = pyHandler.executeScriptAsync();
              System.out.println(String.format("[droid] Return value: %s", output));
              result.success(output);
            } catch (Exception ex) {
              result.error("1", ex.getMessage(), null);
            }
          }
        }).start();

    } else {
      result.notImplemented();
    }
  }

  @Override
  public void onDetachedFromEngine(@NonNull FlutterPluginBinding binding) {
    channel.setMethodCallHandler(null);
  }
}
