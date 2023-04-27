import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';

import 'dart:developer';
import 'dart:io';
import 'dart:ui' as ui;

import 'package:pydroid/pydroid.dart';
import 'package:pydroid_example/face_detection/canvas.dart';

// A widget that displays the picture taken by the user.
// Contains Canvas to draw the bounding box on the image

class DisplayPictureScreen extends StatefulWidget {
  final XFile image;

  const DisplayPictureScreen({Key? key, required this.image}) : super(key: key);

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

  FacePainter? facePainter;

  Future<ui.Image> _convertToImage(String imagePath) async {
    // creates a ui.Image object from a JPG file
    // this type of image is needed to be custom drawn on a canvas
    // issue is it requires async, which won't work in painter

    log("[convertToImage] Start...");
    var bytes = await widget.image.readAsBytes();
    final codec = await ui.instantiateImageCodec(
      bytes,
      targetHeight: 732,
      targetWidth: 401,
    );
    final imageFromFile = (await codec.getNextFrame()).image;
    return imageFromFile;
  }

  void _detectFace() async {
    // final i = await _convertToImage(widget.image.path);  // *** hack
    // facePainter!.image = i;  // ***hack
    // setState(() {
    //   // update w/ face
    // });

    // The image is stored as a file on the device. Use the `Image.file`
    // constructor with the given path to display the image.
    log("[Flutter] Detect face start...");
    var img = widget.image.path;
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
          // facePainter!.face = value; // ***for hack
          _isComputationRunning = false; // unset blocker
        });
      });
    } else {
      log("[Flutter] main.dart - computation blocked (already running)!");
    }
  }

  @override
  void initState() {
    super.initState();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('')),
      body: Column(
        mainAxisAlignment: MainAxisAlignment.start,
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: <Widget>[
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceEvenly,
            children: [
              TextButton(
                onPressed: _detectFace,
                child: const Text(
                  "Detect",
                  textScaleFactor: 1.4,
                ),
              ),
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
              ), // get reference to facePainter so we can update the image object ***hack,
              child: Image.file(
                File(widget.image.path),
                scale: 0.75,
              ),
              // child: const SizedBox.shrink(),
            ),
            decoration: BoxDecoration(
              color: Colors.red,
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
