import 'package:flutter/material.dart';
import 'dart:developer';
import 'dart:async';
import 'dart:io';

import 'package:pydroid/pydroid.dart';

// A widget that displays the picture taken by the user.
// Button to run Python method in background to get facial bounding box
// Canvas to draw the bounding box on the image

class DisplayPictureScreen extends StatefulWidget {
  final String imagePath;

  DisplayPictureScreen({Key? key, required this.imagePath}) : super(key: key);

  @override
  State<DisplayPictureScreen> createState() => _DisplayPictureScreenState();
}

class _DisplayPictureScreenState extends State<DisplayPictureScreen> {
  bool _isComputationRunning = false;

  void _detectFace() async {
    // The image is stored as a file on the device. Use the `Image.file`
    // constructor with the given path to display the image.
    log("[Flutter] Detect face start...");
    var img = widget.imagePath;
    log("Image is located @ $img");

    if (!_isComputationRunning) {
      log("[Flutter] main.dart - Executing script...");
      setState(() {
        _isComputationRunning = true; // set blocker
      });
      var startTime = DateTime.now();
      Pydroid.getFaceForImage(img).then((value) {
        var totalTime = DateTime.now().difference(startTime).inMilliseconds;
        log("[Flutter] Received value '$value'");
        setState(() {
          _isComputationRunning = false; // set blocker
        });
      });
    } else {
      log("[Flutter] main.dart - computation blocked (already running)!");
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Display the Picture')),
      body: Column(
        children: <Widget>[
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceEvenly,
            children: [
              TextButton(
                  onPressed: _detectFace, child: const Text("Detect Face")),
              (_isComputationRunning)
                  ? const CircularProgressIndicator(
                      backgroundColor: Colors.red,
                    )
                  : const SizedBox.shrink(),
            ],
          ),
          Image.file(File(widget.imagePath)),
        ],
      ),
    );
  }
}
