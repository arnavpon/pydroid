package com.jacobsonlab.pydroid;

import com.chaquo.python.Python;
import com.chaquo.python.PyObject;
import com.chaquo.python.PyException;
import com.chaquo.python.android.AndroidPlatform;

import android.content.Context;
import java.util.List;
import java.util.logging.Level;
import java.util.logging.Logger;

public class PythonHandler {

    private Python pyInstance;
    private Logger logger;
    private Context ctx;

    public PythonHandler(Context context) {
        System.out.println("PythonHandler init...");
        // if (logger == null) {  // intialize logger
        //     logger = Logger.getLogger("main");
        //     logger.setLevel(Level.ALL);
        //     logger.log(Level.INFO, "initialized logger!");
        // }
        ctx = context;
        if (!Python.isStarted()) {  // initialize Python instance
            logger.log(Level.INFO, "starting Python instance...");
            Python.start(new AndroidPlatform(context));  // where is the "context"
            pyInstance = Python.getInstance();
        } else {
            System.out.println("Python instance already running!");
        }
    }

    public String test() {
        System.out.println("PythonHandler - test() - top...");
        // Python.start(new AndroidPlatform(ctx));  // where is the "context"
        pyInstance = Python.getInstance();
        PyObject sys = pyInstance.getModule("sys");  // call module
        System.out.println("PythonHandler - test() - mid...");
        String res = sys.get("version").toString();
        System.out.println(res);
        return res;
    }
  
}
