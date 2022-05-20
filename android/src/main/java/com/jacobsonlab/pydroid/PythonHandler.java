package com.jacobsonlab.pydroid;

import com.chaquo.python.Python;
import com.chaquo.python.PyObject;
import com.chaquo.python.PyException;
import com.chaquo.python.android.AndroidPlatform;

import android.content.Context;
import android.os.Bundle;
import android.os.Handler;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;

import java.nio.charset.StandardCharsets;
import java.util.List;

import java.util.concurrent.Executor;

// TODO 
// Consuming python scripts

public class PythonHandler {

    private final Executor executor;  // background thread
    private Python pyInstance;
    private Context ctx;  // app context gives access to filesystem

    public PythonHandler(Context context, Executor executor) {
        System.out.println("PythonHandler init...");
        this.ctx = context;
        this.executor = executor;
        if (!Python.isStarted()) {  // initialize Python instance
            Python.start(new AndroidPlatform(context));  // where is the "context"
            pyInstance = Python.getInstance();
        } else {
            System.out.println("Python instance already running!");
        }
    }

    public String test() {
        System.out.println("PythonHandler - test()...");
        if (!Python.isStarted()) { // initialize Python instance
            System.out.println("PythonHandler instance NOT started...");
            // Python.start(new AndroidPlatform(ctx));  // where is the "context"?
            // no access since initializer is not being called
        }
        pyInstance = Python.getInstance();

        PyObject os = pyInstance.getModule("os");
        System.out.println(os.get("environ").asMap().get("HOME").toString());

        PyObject sys = pyInstance.getModule("sys");  // call module
        System.out.println(sys.get("path").toString());
        String res = sys.get("version").toString();
        System.out.println(res);
        return res;
    }

    public String readFile(String filename) throws IOException {
        if (ctx == null) {
            System.out.println("Context is null!");
            return "";
        } else {
            System.out.println(String.format("[java - readFile] Context EXISTS - reading file 'name=%s'", filename));
        }
        
        System.out.println(ctx.fileList().toString());
        FileInputStream fis = ctx.openFileInput(filename);
        InputStreamReader inputStreamReader = new InputStreamReader(fis);
        StringBuilder stringBuilder = new StringBuilder();
        try (BufferedReader reader = new BufferedReader(inputStreamReader)) {
            String line = reader.readLine();
            while (line != null) {
                stringBuilder.append(line).append("\n");
                line = reader.readLine();
            }
        } catch (IOException e) {
            System.out.println(e.getMessage());
        }
        return stringBuilder.toString();
    }

    public String executeScriptSync() {
        /// Utilizes script.py to run arbitrary code on a separate thread using "exec"

        pyInstance = Python.getInstance();
        PyObject sys = pyInstance.getModule("sys");
        PyObject io = pyInstance.getModule("io");
        PyObject threaderScript = pyInstance.getModule("custom_threader");
        PyObject helpers = pyInstance.getModule("helpers");  // *** to read file
        String interpreterOutput = "";
        try {
            PyObject textOutputStream = io.callAttr("StringIO");
            sys.put("stdout", textOutputStream);

            // obtain code to execute as string, simply read in module? or as text for now?
            String code = "print(2 * 3 ** 2)";  // read in code as string, need this?...

            threaderScript.callAttrThrows("mainTextCode", code);
            interpreterOutput = textOutputStream.callAttr("getvalue").toString();   
        } catch (IOException e) {
            System.out.println(e.getMessage());
        } catch (PyException e) {
            // error in python code
            interpreterOutput = e.getMessage().toString();
        } catch (Throwable throwable) {
            // java error
            throwable.printStackTrace();
        }
        System.out.println(interpreterOutput);
        return interpreterOutput;
    }

    public void executeScriptAsync(int iterations, int modelSize, final PythonCallback callback) {
        /// Runs python script on background Android thread

        System.out.println("[executeAsync] thread: " + Thread.currentThread().getName());
        /// Background thread
        executor.execute(new Runnable() {
            @Override
            public void run() { // single abstract method interface
                // make synchronous python call in this background thread
                System.out.println("[executeAsync - run] thread: " + Thread.currentThread().getName());
                pyInstance = Python.getInstance();
                PyObject sys = pyInstance.getModule("sys");
                PyObject io = pyInstance.getModule("io");
                String interpreterOutput = "";  // object returned in callback
                try {
                    PyObject textOutputStream = io.callAttr("StringIO");
                    sys.put("stdout", textOutputStream);
                    PyObject lr = pyInstance.getModule("lin_reg");
                    Float avg = lr.callAttr("average_performance", iterations, modelSize).toFloat();
                    System.out.println(String.format("\n[java] executeScript - Average Value: [%f]", avg));
                    interpreterOutput = String.format("%f", avg);
                    // interpreterOutput = textOutputStream.callAttr("getvalue").toString();
                    System.out.println("[java] Python Script output: " + interpreterOutput);
                } catch (PyException e) {
                    // error in python code - return error msg through callback
                    interpreterOutput = e.getMessage();
                } catch (Throwable throwable) {
                    // error in java code - return error msg through callback
                    throwable.printStackTrace();
                    interpreterOutput = "[java exception] " + throwable.getMessage();
                }
                callback.onComplete(interpreterOutput);
            }
        });
    }
  
}
