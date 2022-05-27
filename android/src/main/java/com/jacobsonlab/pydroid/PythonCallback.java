package com.jacobsonlab.pydroid;
import java.util.HashMap;

interface PythonCallback {
    void onComplete(HashMap<String, String> result);
}