/*
Video screen for taking face video for the HR estimation.
Based on this example: 
https://github.com/flutter/plugins/blob/main/packages/camera/camera/example/lib/main.dart
*/

import 'package:flutter/material.dart';
import 'dart:developer';
import 'dart:async';
import 'dart:io';

import 'package:flutter/foundation.dart';
import 'package:flutter/scheduler.dart';
import 'package:video_player/video_player.dart';

import 'package:camera/camera.dart';
import 'package:pydroid_example/face_detection/display_picture_screen.dart';

// A screen that allows users to take a picture using a given camera.
class TakeVideoScreen extends StatefulWidget {
  const TakeVideoScreen({
    Key? key,
    required this.camera,
  }) : super(key: key);

  final CameraDescription camera;

  @override
  TakeVideoScreenState createState() => TakeVideoScreenState();
}

class TakeVideoScreenState extends State<TakeVideoScreen> 
    with WidgetsBindingObserver, TickerProviderStateMixin {

  late CameraController _controller;
  late Future<void> _initializeControllerFuture;
  XFile? videoFile;
  VideoPlayerController? videoController;
  VoidCallback? videoPlayerListener;
  bool enableAudio = false;
  bool currentlyRecording = false;
  bool videoTaken = false;
  double _minAvailableExposureOffset = 0.0;
  double _maxAvailableExposureOffset = 0.0;
  double _currentExposureOffset = 0.0;
  late AnimationController _flashModeControlRowAnimationController;
  late Animation<double> _flashModeControlRowAnimation;
  late AnimationController _exposureModeControlRowAnimationController;
  late Animation<double> _exposureModeControlRowAnimation;
  late AnimationController _focusModeControlRowAnimationController;
  late Animation<double> _focusModeControlRowAnimation;
  double _minAvailableZoom = 1.0;
  double _maxAvailableZoom = 1.0;
  double _currentScale = 1.0;
  double _baseScale = 1.0;

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addObserver(this);

    _controller = CameraController(
      widget.camera, // specify camera (e.g. front or back)
      ResolutionPreset.medium, // resolution
    );

    _flashModeControlRowAnimationController = AnimationController(
      duration: const Duration(milliseconds: 300),
      vsync: this,
    );
    _flashModeControlRowAnimation = CurvedAnimation(
      parent: _flashModeControlRowAnimationController,
      curve: Curves.easeInCubic,
    );
    _exposureModeControlRowAnimationController = AnimationController(
      duration: const Duration(milliseconds: 300),
      vsync: this,
    );
    _exposureModeControlRowAnimation = CurvedAnimation(
      parent: _exposureModeControlRowAnimationController,
      curve: Curves.easeInCubic,
    );
    _focusModeControlRowAnimationController = AnimationController(
      duration: const Duration(milliseconds: 300),
      vsync: this,
    );
    _focusModeControlRowAnimation = CurvedAnimation(
      parent: _focusModeControlRowAnimationController,
      curve: Curves.easeInCubic,
    );

    // Next, initialize the controller. This returns a Future.
    _initializeControllerFuture = _controller.initialize();
  }

  @override
  void dispose() {
    WidgetsBinding.instance.removeObserver(this);
    _flashModeControlRowAnimationController.dispose();
    _exposureModeControlRowAnimationController.dispose();

    // Dispose of the controller when the widget is disposed.
    _controller.dispose();
    super.dispose();
  }

  Future<void> startVideoRecording() async {
    final CameraController? cameraController = _controller;

    if (cameraController == null || !cameraController.value.isInitialized) {
      return;
    }

    if (cameraController.value.isRecordingVideo) {
      // A recording is already started, do nothing.
      return;
    }

    try {
      await cameraController.startVideoRecording();
    } on CameraException catch (e) {
      // _showCameraException(e);
      return;
    }
  }

  Future<XFile?> stopVideoRecording() async {
    final CameraController? cameraController = _controller;

    if (cameraController == null || !cameraController.value.isRecordingVideo) {
      return null;
    }

    try {
      return cameraController.stopVideoRecording();
    } on CameraException catch (e) {
      // _showCameraException(e);
      return null;
    }
  }

  void onVideoRecordButtonPressed() {
    print('starting');
    currentlyRecording = true;
    startVideoRecording().then((_) {
      if (mounted) {
        setState(() {});
      }
    });
  }

  void onStopButtonPressed() {
    print("stopping");
    currentlyRecording = false;
    stopVideoRecording().then((XFile? file) {
      if (mounted) {
        setState(() {});
      }
      if (file != null) {
        videoFile = file;
        videoTaken = true;
        _startVideoPlayer();
      }
    });
  }

  Future<void> _startVideoPlayer() async {
    print('\n\n\n\n\nIN VID\N\N\N\N\N\N');
    if (videoFile == null) {
      print('RETUNN');
      return;
    }
    
    print('\n\n\n\n\n\nat this final\n\n\n\n\n\n\n\n');
    final VideoPlayerController vController = kIsWeb
        ? VideoPlayerController.network(videoFile!.path)
        : VideoPlayerController.file(File(videoFile!.path));
  
    print('\n\n\n\n\nGot to video player listene\n\n\n\n\n\n');
    videoPlayerListener = () {
      if (videoController != null && videoController!.value.size != null) {
        // Refreshing the state to update video player with the correct ratio.
        if (mounted) {
          setState(() {});
        }
        videoController!.removeListener(videoPlayerListener!);
      }
    };

    print('here we go');
    vController.addListener(videoPlayerListener!);
    print('after1');
    await vController.setLooping(true);
    print('after2');
    await vController.initialize();
    print('after3');
    await videoController?.dispose();
    print('after4');
    if (mounted) {
      setState(() {
        videoController = vController;
      });
    }

    print('after5');
    await vController.play();
    print('got through');
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Take a picture')),
      // You must wait until the controller is initialized before displaying the
      // camera preview. Use a FutureBuilder to display a loading spinner until the
      // controller has finished initializing.
      body: FutureBuilder<void>(
        future: _initializeControllerFuture,
        builder: (context, snapshot) {
          // if (snapshot.connectionState == ConnectionState.done) {
          if (videoFile != null && videoTaken) {
            // If the Future is complete, display the preview.
            return _thumbnailWidget();
          } else if (snapshot.connectionState == ConnectionState.done) {
            return CameraPreview(_controller);
          } else {
            // Otherwise, display a loading indicator.
            return const Center(child: CircularProgressIndicator());
          }
        },
      ),
      floatingActionButton: FloatingActionButton(
        onPressed: currentlyRecording ? onStopButtonPressed : onVideoRecordButtonPressed,
        child: const Icon(Icons.videocam)
      )
    );
  }

  Widget _thumbnailWidget() {
    final VideoPlayerController? localVideoController = videoController;

    if (localVideoController == null) print("IT'S NULL");
    else print("IT ISN'T NULL");
    return Expanded(
      child: Align(
        alignment: Alignment.centerRight,
        child: Row(
          mainAxisSize: MainAxisSize.min,
          children: <Widget>[
            if (localVideoController == null)
              Container()
            else
              SizedBox(
                width: 64.0,
                height: 64.0,
                child: Container(
                        decoration: BoxDecoration(
                            border: Border.all(color: Colors.pink)),
                        child: Center(
                          child: AspectRatio(
                              aspectRatio:
                                  localVideoController.value.size != null
                                      ? localVideoController.value.aspectRatio
                                      : 1.0,
                              child: VideoPlayer(localVideoController)),
                        ),
                      ),
              ),
          ],
        ),
      ),
    );
  }
}