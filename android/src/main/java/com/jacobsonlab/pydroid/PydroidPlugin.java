package com.jacobsonlab.pydroid;

import androidx.annotation.NonNull;
import android.os.Handler;
import androidx.core.os.HandlerCompat;
import android.os.Looper;

import java.util.HashMap;
import java.util.concurrent.Executor;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

import android.app.Activity;

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

  private ExecutorService executorService;  // thread manager
  private Handler mainThreadHandler;  // save instance of main thread for quick ref

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
    this.channel = new MethodChannel(flutterPluginBinding.getBinaryMessenger(), channelName);
    this.channel.setMethodCallHandler(this);
    this.context = flutterPluginBinding.getApplicationContext();

    // set up thread pool for background tasks, initialize the pool ONCE only! Is this function called only a single time??? Verify
    this.executorService = Executors.newSingleThreadExecutor();  // only need 1 thread
    this.pyHandler = new PythonHandler(context, executorService);  // init python handler w/ executor
    this.mainThreadHandler = HandlerCompat.createAsync(Looper.getMainLooper());
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

    } else if (call.method.equals("executeInBackground")) {
      System.out.println("[java] Generic execute in background function...");
      System.out.println("Current thread: " + Thread.currentThread().getName());

      String pyScript = call.argument(PythonHandler.KEY_SCRIPT);
      System.out.println(call.arguments().toString());
      HashMap<String, Object> allArguments = call.arguments();
      System.out.println("[java] Call arguments " + allArguments);
      // convert to JSON to pass to python?
      try {
        pyHandler.executePyScriptAsync(pyScript, allArguments, new PythonCallback() {
          @Override
          public void onComplete(HashMap<String, String> pyOutput) {
            System.out.println(String.format("[java] Return dict: %s", pyOutput));
            mainThreadHandler.post(new Runnable() {
              // return result on UI (main) thread else app crashes!
              @Override
              public void run() {
                System.out.println("inside run method");
                // Generate alert to Dart main thread that platform method has returned its output through callback (this is async, platform method call doesn't return anything)
                if (pyOutput.containsKey(PythonHandler.KEY_OUTPUT_ERROR)) {
                  // error occurred
                  System.out.println("returning ERROR");
                  String errorMsg = pyOutput.get(PythonHandler.KEY_OUTPUT_ERROR);
                  result.error("00", errorMsg, null);
                } else {
                  System.out.println("returning SUCCESS");
                  String resultAsJSON = pyOutput.get(PythonHandler.KEY_OUTPUT_VALUE);  // object comes back as JSON
                  result.success(resultAsJSON);
                }
              }
            });
          }
        });
      } catch (Exception ex) {
        result.error("1", ex.getMessage(), null);
      }

    } else if (call.method.equals("executeScriptLR")) {
      int iterations = call.argument("iterations");
      int modelSize = call.argument("modelSize");
      System.out.println("[droid] executeInBackground...");
      System.out.println("Current thread: " + Thread.currentThread().getName());
      try {
        pyHandler.executeScriptLRAsync(iterations, modelSize, new PythonCallback() {
          @Override
          public void onComplete(HashMap<String, String> pyOutput) {
            System.out.println(String.format("[droid] Return value: %s", pyOutput));
            mainThreadHandler.post(new Runnable() { // return result on UI thread
              // if result methods are called outside of main, crashes app!
              @Override
              public void run() {
                result.success(pyOutput);  // generates alert to Dart main thread that platform method has returned its output; this is async, b/c platform method call doesn't return anything
              }
            });
          }
        });
      } catch (Exception ex) {
        result.error("1", ex.getMessage(), null);
      }

    } else if (call.method.equals("faceForImage")) { 
      // executes script to get bounding box for image in background
      String pyScript = call.argument(PythonHandler.KEY_SCRIPT);

    } else {
      result.notImplemented();
    }
  }

  @Override
  public void onDetachedFromEngine(@NonNull FlutterPluginBinding binding) {
    channel.setMethodCallHandler(null);
  }
}
