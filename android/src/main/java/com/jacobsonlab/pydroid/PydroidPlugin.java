package com.jacobsonlab.pydroid;

import androidx.annotation.NonNull;
import android.os.Handler;
import androidx.core.os.HandlerCompat;
import android.os.Looper;

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
      int iterations = call.argument("iterations");
      int modelSize = call.argument("modelSize");
      System.out.println("[droid] executeInBackground...");
      System.out.println("Current thread: " + Thread.currentThread().getName());
            try {
              pyHandler.executeScriptAsync(iterations, modelSize, new PythonCallback() {
                @Override
                public void onComplete(String pyOutput) {
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

    } else {
      result.notImplemented();
    }
  }

  @Override
  public void onDetachedFromEngine(@NonNull FlutterPluginBinding binding) {
    channel.setMethodCallHandler(null);
  }
}
