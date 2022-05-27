import 'package:flutter/material.dart';
import 'dart:developer';
import 'dart:io';

import 'package:pydroid/pydroid.dart';
import 'package:pydroid_example/face_detection/canvas.dart';

// A widget that displays the picture taken by the user.
// Button to run Python method in background to get facial bounding box
// Canvas to draw the bounding box on the image

class DisplayPictureScreen extends StatefulWidget {
  final String imagePath;

  const DisplayPictureScreen({Key? key, required this.imagePath})
      : super(key: key);

  @override
  State<DisplayPictureScreen> createState() => _DisplayPictureScreenState();
}

class _DisplayPictureScreenState extends State<DisplayPictureScreen> {
  bool _isComputationRunning = false;
  bool _isInitialLoad = true;
  Color get _borderColor =>
      ((_face == Rect.zero) ? Colors.red : Colors.green.shade200);
  Rect _face = Rect.zero;
  Rect _forehead = Rect.zero;

  void _detectFace() {
    // The image is stored as a file on the device. Use the `Image.file`
    // constructor with the given path to display the image.
    log("[Flutter] Detect face start...");
    var img = widget.imagePath;
    _isInitialLoad = false; // set indicator so border color is set correctly
    log("Image is located @ $img");

    if (!_isComputationRunning) {
      log("[Flutter] main.dart - Executing script...");
      setState(() {
        _isComputationRunning = true; // set blocker
      });
      var startTime = DateTime.now();
      Pydroid.getFaceForImage(img).then((value) {
        var totalTime = DateTime.now().difference(startTime).inMilliseconds;
        log("[Flutter] Received value '$value' in $totalTime milliseconds");
        setState(() {
          _face = value;
          _isComputationRunning = false; // unset blocker
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
          Container(
            child: CustomPaint(
              foregroundPainter: FacePainter(
                context,
                _face,
                _forehead,
              ),
              child: Image.file(File(widget.imagePath)),
            ),
            decoration: BoxDecoration(
              border: Border.all(
                  color: _isInitialLoad ? Colors.transparent : _borderColor,
                  width: 5),
            ),
          ),
        ],
      ),
    );
  }
}
