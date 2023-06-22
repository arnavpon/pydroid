import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import 'package:gallery_saver/gallery_saver.dart';
import 'package:pydroid/pydroid.dart';
import 'package:pydroid_example/face_detection/hr_screen.dart';

class VideoPage extends StatefulWidget {
  const VideoPage({Key? key}) : super(key: key);

  @override
  _VideoPageState createState() => _VideoPageState();
}

class _VideoPageState extends State<VideoPage> {
  
  bool _isLoading = true;
  bool _isRecording = false;

  // initialize face box to a box with no area
  final List<dynamic> _faces = [{'x1': 0, 'y1': 0, 'x2': 0, 'y2': 0}];
  List<dynamic> _hr = [];
  
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

      print('[Dart] Analyzing video...');
      final value = await Pydroid.analyzeVideo(file.path);
      setState(() => _hr = value[0]);

      // Navigate to the new screen and pass the _hr value
      Navigator.push(
        context,
        MaterialPageRoute(builder: (context) => HRScreen(hr: _hr)),
      );

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
            face: _faces[0],
            aspectRatio: 1,
            color: Colors.cyan
          ),
          if (_faces.length > 1) cameraOverlay(
            face: _faces[1],
            aspectRatio: 1,
            color: Colors.red
          ),
          if (_faces.length > 2) cameraOverlay(
            face: _faces[2],
            aspectRatio: 1,
            color: Colors.green
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
            decoration: BoxDecoration(border: Border.all(color: color)),
          )
        ]
      );
    });
  }
}
