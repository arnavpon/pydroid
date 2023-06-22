/*
Video screen for taking face video for the HR estimation.
Based on this example: 
https://github.com/flutter/plugins/blob/main/packages/camera/camera/example/lib/main.dart
*/
import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import 'package:pydroid_example/face_detection/show_video.dart';
import 'package:pydroid_example/face_detection/canvas.dart';

class CameraPage extends StatefulWidget {
  const CameraPage({Key? key}) : super(key: key);

  @override
  _CameraPageState createState() => _CameraPageState();
}

class _CameraPageState extends State<CameraPage> {
  bool _isLoading = true;
  bool _isRecording = false;
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
    final cameras = await availableCameras();
    final front = cameras.firstWhere(
        (camera) => camera.lensDirection == CameraLensDirection.front);
    _cameraController = CameraController(front, ResolutionPreset.max);
    await _cameraController.initialize();
    setState(() => _isLoading = false);
  }

  _recordVideo() async {
    if (_isRecording) {
      final file = await _cameraController.stopVideoRecording();
      setState(() => _isRecording = false);
      final route = MaterialPageRoute(
        fullscreenDialog: true,
        builder: (_) => VideoPage(filePath: file.path),
      );
      Navigator.push(context, route);
    } else {
      await _cameraController.prepareForVideoRecording();
      await _cameraController.startVideoRecording();
      setState(() => _isRecording = true);
    }
  }

  @override
  Widget build(BuildContext context) {
    if (_isLoading) {
      return Container(
        color: Colors.white,
        child: const Center(
          child: CircularProgressIndicator(),
        ),
      );
    } else {
      return Center(
        child: Stack(
          alignment: Alignment.bottomCenter,
          children: [
            CameraPreview(_cameraController),
            // CustomPaint(
            //   foregroundPainter: FacePainter(
            //     context,
            //     _face,
            //     _forehead,
            //   ), // get reference to facePainter so we can update the image object ***hack,
            //   child: const Text(""),
            //   // child: const SizedBox.shrink(),
            // ),
          ],

          // children: [
          //   CameraPreview(_cameraController),
          //   // Padding(
          //   //   padding: const EdgeInsets.all(25),
          //   //   child: FloatingActionButton(
          //   //     backgroundColor: Colors.red,
          //   //     child: Icon(_isRecording ? Icons.stop : Icons.circle),
          //   //     onPressed: () => _recordVideo(),
          //   //   ),
          //   // ),
          //   Positioned(
          //     left: _position['x'],
          //     top: _position['y'],
          //     child: InkWell(
          //       onTap: () {
          //         // When the user taps on the rectangle, it will disappear
          //         setState(() {
          //           _isRectangleVisible = false;
          //         });
          //       },
          //       child: Container(
          //         width: _position['w'],
          //         height: _position['h'],
          //         decoration: BoxDecoration(
          //           border: Border.all(
          //             width: 2,
          //             color: Colors.blue,
          //           ),
          //         ),
          //         child: Align(
          //           alignment: Alignment.topLeft,
          //           child: Container(
          //             color: Colors.blue,
          //             child: Text(
          //               'hourse -71%',
          //               style: TextStyle(color: Colors.white),
          //             ),
          //           ),
          //         ),
          //       ),
          //     ),
          //   ),
          // ],
        ),
      );
    }
  }
}
