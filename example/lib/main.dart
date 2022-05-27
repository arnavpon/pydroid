import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'dart:async';
import 'dart:developer';

import 'package:camera/camera.dart';
import 'package:pydroid/pydroid.dart';
import 'package:pydroid_example/face_detection/take_picture_screen.dart';

void main() {
  WidgetsFlutterBinding.ensureInitialized();
  runApp(MaterialApp(title: "Pydroid Example", home: MyApp()));
}

class MyApp extends StatefulWidget {
  @override
  _MyAppState createState() => _MyAppState();
}

class _MyAppState extends State<MyApp> {
  String? _platformVersion = 'Unknown';
  int _hitCount = 0;
  bool _isComputationRunning = false;
  int _nIterations = 10;
  int _modelSize = 10000;
  int _computationTime = 0;
  String? _returnValue = "0";

  late TextEditingController _controllerN;
  late TextEditingController _controllerMS;

  @override
  void initState() {
    super.initState();
    initPlatformState();
    _controllerN = TextEditingController();
    _controllerMS = TextEditingController();
  }

  // Platform messages are asynchronous, so we initialize in an async method.
  Future<void> initPlatformState() async {
    String? platformVersion;
    try {
      platformVersion = await Pydroid.platformVersion;
    } on PlatformException {
      platformVersion = 'Failed to get platform version.';
    }

    if (!mounted) return;

    setState(() {
      _platformVersion = platformVersion;
    });
  }

  void _runTest() async {
    log("[Flutter] _runTest");
    // Pydroid.runTest().then((value) => log("Returned string of length $value"));

    // open screen to take picture
    // display image in view
    final cameras = await availableCameras();
    log(cameras.toString());
    final firstCamera = cameras.last;
    Navigator.push(
        context,
        MaterialPageRoute(
            builder: (context) => TakePictureScreen(camera: firstCamera)));
  }

  void _executeScriptLR() {
    if (!_isComputationRunning) {
      log("[Flutter] main.dart - Executing script...");
      setState(() {
        _isComputationRunning = true; // set blocker
      });
      var startTime = DateTime.now();
      Pydroid.executeLinRegInBackground(_nIterations, _modelSize).then((value) {
        var totalTime = DateTime.now().difference(startTime).inMilliseconds;
        log("[Flutter] Received value '$value'");
        setState(() {
          _returnValue = value;
          _computationTime = totalTime;
          _isComputationRunning = false; // unset blocker
        });
      });
    } else {
      log("[Flutter] main.dart - computation blocked (already running)!");
    }
  }

  void _incrementCounter() {
    setState(() {
      _hitCount++;
    });
  }

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      home: Scaffold(
        backgroundColor: Colors.white,
        body: Center(
          child: Column(
            mainAxisAlignment: MainAxisAlignment.spaceEvenly,
            children: <Widget>[
              Text('Running on: $_platformVersion\n'),
              TextButton(
                  onPressed: _executeScriptLR, child: const Text("Run Test")),
              Row(
                mainAxisAlignment: MainAxisAlignment.center,
                children: <Widget>[
                  IconButton(
                      onPressed: _incrementCounter,
                      iconSize: 40,
                      icon: const Icon(Icons.add)),
                  Text("Hit Count: $_hitCount"),
                ],
              ),
              Row(
                mainAxisAlignment: MainAxisAlignment.center,
                children: <Widget>[
                  Expanded(
                    child: TextField(
                      controller: _controllerN,
                      decoration: const InputDecoration(
                        border: OutlineInputBorder(),
                        labelText: '# of Iterations',
                      ),
                      onChanged: (text) {
                        setState(() {
                          try {
                            _nIterations = int.parse(text);
                          } catch (e) {
                            log("[flutter] Parser: value error");
                          }
                        });
                      },
                    ),
                  ),
                  Expanded(
                    child: TextField(
                      controller: _controllerMS,
                      decoration: const InputDecoration(
                        border: OutlineInputBorder(),
                        labelText: 'Model Size',
                      ),
                      onChanged: (text) {
                        setState(() {
                          try {
                            _modelSize = int.parse(text);
                          } catch (e) {
                            log("[flutter] Parser: value error");
                          }
                        });
                      },
                    ),
                  ),
                ],
              ),
              Row(
                mainAxisAlignment: MainAxisAlignment.center,
                children: <Widget>[
                  (_isComputationRunning)
                      ? const CircularProgressIndicator(
                          backgroundColor: Colors.red,
                        )
                      : const SizedBox.shrink(),
                  TextButton(
                    onPressed: _runTest,
                    child: const Text("Take picture"),
                  ),
                ],
              ),
              Text("Total Computation Time: $_computationTime (ms)"),
              Text("Avg Time per Iteration (RV): $_returnValue (ms)"),
            ],
          ),
        ),
      ),
    );
  }
}
