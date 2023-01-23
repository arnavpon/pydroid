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
  bool _isRectangleVisible = true;
  Map<String, dynamic> _face = {'x1': 0, 'y1': 0, 'x2': 0, 'y2': 0};
  Rect _forehead = Rect.zero;
  
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
    final front = cameras.firstWhere((camera) => camera.lensDirection == CameraLensDirection.front);
    _cameraController = CameraController(front, ResolutionPreset.medium);
    await _cameraController.initialize();
    setState(() => _isLoading = false);
  }

  _recordVideo() async {
    if (_isRecording) {
      final file = await _cameraController.stopVideoRecording();
      setState(() => _isRecording = false);
      
      Pydroid.analyzeVideo(file.path).then((value) {
          setState(() {
            _face = value;
          });
          print('Face is now');
          print(value);
        });

      // final route = MaterialPageRoute(
      //   fullscreenDialog: true,
      //   builder: (_) => VideoPage(filePath: file.path),
      // );
      // Navigator.push(context, route);
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
              child: Stack(fit: StackFit.expand, alignment: Alignment.bottomCenter,
              children: [
                CameraPreview(_cameraController),
                cameraOverlay(
                    face: _face, aspectRatio: 1, color: Colors.transparent),
                Align(
                  alignment: Alignment.bottomCenter,
                  child: Padding(
                  padding: const EdgeInsets.all(25),
                  child: FloatingActionButton(
                    backgroundColor: Colors.red,
                    child: Icon(_isRecording ? Icons.stop : Icons.circle),
                    onPressed: () => _recordVideo(),
                  ),
                )),
              ]));
}

  Widget cameraOverlay({required Map<String, dynamic> face, required double aspectRatio, required Color color}) {
      return LayoutBuilder(builder: (context, constraints) {
        double parentAspectRatio = constraints.maxWidth / constraints.maxHeight;
        double horizontalPadding;
        double verticalPadding;

        // if (parentAspectRatio < aspectRatio) {
        //   print('this aspect ration');
        //   horizontalPadding = padding * 2;
        //   verticalPadding = (constraints.maxHeight -
        //           ((constraints.maxWidth - 2 * padding) / aspectRatio)) /
        //       2;
        // } else {
        //   print('no this one');
        //   verticalPadding = padding;
        //   horizontalPadding = (constraints.maxWidth -
        //           ((constraints.maxHeight - 2 * padding) * aspectRatio)) /
        //       2;
        // }
        return Stack(fit: StackFit.expand, children: [
          // Align(
          //     // alignment: Alignment.centerLeft,
          //     child: Container(width: horizontalPadding, color: color)),
          // Align(
          //     // alignment: Alignment.centerRight,
          //     child: Container(width: horizontalPadding * 2, color: color)),
          // Align(
          //     // alignment: Alignment.topCenter,
          //     child: Container(
          //         margin: EdgeInsets.only(
          //             left: horizontalPadding, right: horizontalPadding),
          //         height: verticalPadding,
          //         color: color)),
          // Align(
          //     // alignment: Alignment.bottomCenter,
          //     child: Container(
          //         margin: EdgeInsets.only(
          //             left: horizontalPadding, right: horizontalPadding),
          //         height: verticalPadding,
          //         color: color)),
          Container(
            // margin: EdgeInsets.symmetric(
            //     horizontal: horizontalPadding, vertical: verticalPadding),
            margin: EdgeInsets.only(
              left: face['x1'].toDouble(),
              right: (MediaQuery.of(context).size.width - face['x2']).toDouble(),
              top: face['y1'].toDouble(),
              bottom: face['y2'].toDouble()
            ),
            decoration: BoxDecoration(border: Border.all(color: Colors.cyan)),
          )
        ]);
      });
    }

  // @override
  // Widget build(BuildContext context) {
  //   if (_isLoading) {
  //     return Container(
  //       color: Colors.white,
  //       child: const Center(
  //         child: CircularProgressIndicator(),
  //       ),
  //     );
  //   } else {
  //     return Container(
  //           child: Stack(
  //             alignment: Alignment.bottomCenter,
  //             children: [
  //               CustomPaint(
  //                 foregroundPainter: FacePainter(
  //                   context,
  //                   _face,
  //                   _forehead,
  //                 ), // get reference to facePainter so we can update the image object ***hack,
  //                 child: CameraPreview(_cameraController),
  //                 // child: const SizedBox.shrink(),
  //               ),
  //               Padding(
  //                 padding: const EdgeInsets.all(25),
  //                 child: FloatingActionButton(
  //                   backgroundColor: Colors.red,
  //                   child: Icon(_isRecording ? Icons.stop : Icons.circle),
  //                   onPressed: () => _recordVideo(),
  //                 ),
  //               ),
  //             ],
  //             // decoration: BoxDecoration(
  //             //     color: Colors.red,
  //             //     border: Border.all(
  //             //         color: Colors.blue,
  //             //         width: 5),
  //             //   ),
  //           )
  //     );
  //     // return Material(
  //     //   child: Stack(
  //     //     alignment: Alignment.bottomCenter,
  //     //     children: [
  //     //       CameraPreview(_cameraController),
  //     //       Padding(
  //     //         padding: const EdgeInsets.all(25),
  //     //         child: FloatingActionButton(
  //     //           backgroundColor: Colors.red,
  //     //           child: Icon(_isRecording ? Icons.stop : Icons.circle),
  //     //           onPressed: () => _recordVideo(),
  //     //         ),
  //     //       ),
  //     //       Positioned(
  //     //         left: _position['x'],
  //     //         top: _position['y'],
  //     //         child: InkWell(
  //     //           onTap: () {
  //     //             // When the user taps on the rectangle, it will disappear
  //     //             setState(() {
  //     //               _isRectangleVisible = false;
  //     //             });
  //     //           },
  //     //           child: Container(
  //     //             width: _position['w'],
  //     //             height: _position['h'],
  //     //             decoration: BoxDecoration(
  //     //               border: Border.all(
  //     //                 width: 2,
  //     //                 color: Colors.blue,
  //     //               ),
  //     //             ),
  //     //             child: Align(
  //     //               alignment: Alignment.topLeft,
  //     //               child: Container(
  //     //                 color: Colors.blue,
  //     //                 child: Text(
  //     //                   'hourse -71%',
  //     //                   style: TextStyle(color: Colors.white),
  //     //                 ),
  //     //               ),
  //     //             ),
  //     //           ),
  //     //         ),
  //     //       ),
  //     //     ],
  //     //   ),
  //     // );
  //   }
  // }
}
