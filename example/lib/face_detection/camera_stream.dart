import 'dart:io';
import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import 'package:pydroid_example/face_detection/show_video.dart';
import 'package:path_provider/path_provider.dart';
import 'package:gallery_saver/gallery_saver.dart';
import 'package:pydroid/pydroid.dart';
import 'package:pydroid_example/face_detection/canvas.dart';

class VideoPage extends StatefulWidget {
  const VideoPage({Key? key}) : super(key: key);

  @override
  _VideoPageState createState() => _VideoPageState();
}

class _VideoPageState extends State<VideoPage> {
  
  bool _isLoading = true;
  bool _isRecording = false;

  // initialize face box to a box with no area
  Map<String, dynamic> _face = {'x1': 0, 'y1': 0, 'x2': 0, 'y2': 0};
  
  // CameraController initialized later
  late CameraController _cameraController;
  
  @override
  void dispose() {
    _cameraController.dispose();
    super.dispose();
  }

  @override
  void initState() {
    super.initState();
    _initCamera();
  }

  _initCamera() async {

    // get camera
    final cameras = await availableCameras();
    final front = cameras.firstWhere((camera) => camera.lensDirection == CameraLensDirection.front);
    
    // initialize CameraController
    _cameraController = CameraController(front, ResolutionPreset.medium);
    await _cameraController.initialize();
    
    setState(() => _isLoading = false);
  }

  _recordVideo() async {

    // if we're already recording, then stop recording and save the video
    if (_isRecording) {

      // collect local path for the video recording and update state 
      // that recording is finished
      final file = await _cameraController.stopVideoRecording();
      setState(() => _isRecording = false);
      
      // get bounding box
      // NOTE: For now, this is just getting the bounding box for the first
      //      frame and overlaying just that box onto the camera preview
      final value = await Pydroid.analyzeVideo(file.path);
      setState(() => _face = value[0]);

    // otherwise we start recording
    } else { 
      await _cameraController.prepareForVideoRecording();
      await _cameraController.startVideoRecording();
      setState(() => _isRecording = true);
    }
  }

  @override
  Widget build(BuildContext context) {
    return AspectRatio(
      aspectRatio: _cameraController.value.aspectRatio,
      child: Stack(
        fit: StackFit.expand,
        alignment: Alignment.bottomCenter,
        children: [
          CameraPreview(_cameraController),
          cameraOverlay(
            face: _face,
            aspectRatio: 1,
            color: Colors.transparent
          ),
          Align(
            alignment: Alignment.bottomCenter,
            child: Padding(
              padding: const EdgeInsets.all(25),
              child: FloatingActionButton(
                backgroundColor: Colors.red,
                child: Icon(_isRecording ? Icons.stop : Icons.circle),
                onPressed: () => _recordVideo(),
              ),
            )
          ),
        ]
      )
    );
  }

  Widget cameraOverlay({required Map<String, dynamic> face, required double aspectRatio, required Color color}) {
    return LayoutBuilder(builder: (context, constraints) {
      return Stack(
        fit: StackFit.expand,
        children: [
          Container(
            margin: EdgeInsets.only(
              left: face['x1'].toDouble(),
              right: (MediaQuery.of(context).size.width - face['x2']).toDouble(),
              top: face['y1'].toDouble(),
              bottom: face['y2'].toDouble()
            ),
            decoration: BoxDecoration(border: Border.all(color: Colors.cyan)),
          )
        ]
      );
    });
  }
}
