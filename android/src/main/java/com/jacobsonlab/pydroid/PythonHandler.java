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

    public List<PyObject> test() {
        System.out.println("PythonHandler - test() - top...");
        // Python.start(new AndroidPlatform(ctx));  // where is the "context"
        // pyInstance = Python.getInstance();

        PyObject os = pyInstance.getModule("os");  // call module
        System.out.println("PythonHandler - test() - mid...");
        List<PyObject> res = os.callAttr("listDir").asList();
        System.out.println(res.toString());
        return res;
    }
  
}
