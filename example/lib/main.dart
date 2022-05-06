import 'dart:developer';

import 'package:flutter/material.dart';
import 'dart:async';

import 'package:flutter/services.dart';
import 'package:pydroid/pydroid.dart';

void main() {
  WidgetsFlutterBinding.ensureInitialized();
  runApp(MyApp());
}

class MyApp extends StatefulWidget {
  @override
  _MyAppState createState() => _MyAppState();
}

class _MyAppState extends State<MyApp> {
  String? _platformVersion = 'Unknown';

  @override
  void initState() {
    super.initState();
    initPlatformState();
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

  void _runTest() {
    print("[Flutter] _runTest");
    Pydroid.runTest().then((value) => log("Returned string of length $value"));
  }

  void _onKeyUp(int key) {
    print("key up:$key");
    Pydroid.onKeyUp(key).then((value) => print(value));
  }

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      home: Scaffold(
        backgroundColor: Color.fromARGB(255, 250, 30, 0),
        body: Center(
          child: Column(
            mainAxisAlignment: MainAxisAlignment.spaceEvenly,
            children: <Widget>[
              Text('Running on: $_platformVersion\n'),
              TextButton(onPressed: _runTest, child: Text("Run Test")),
            ],
          ),
        ),
      ),
    );
  }
}
