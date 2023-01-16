import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:camera/src/camera_controller.dart';

class CameraScreen extends StatefulWidget {
  @override
  _CameraScreenState createState() => _CameraScreenState();
}

class _CameraScreenState extends State<CameraScreen> {
  late CameraController _controller;
  late List<CameraDescription> _cameras;
  bool _isStreaming = false;

  @override
  void initState() {
    super.initState();
    _initializeCamera();
  }

  Future<void> _initializeCamera() async {
    _cameras = await availableCameras();
    _controller = CameraController(_cameras.last, ResolutionPreset.medium);

    _controller.initialize().then((_) {
      if (!mounted) {
        return;
      }
      setState(() {});
    });
  }

  @override
  void dispose() {
    _controller?.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    if (!_controller.value.isInitialized) {
      return Container();
    }
    return Scaffold(
      appBar: AppBar(
        title: Text("Camera"),
      ),
      body: Column(
        children: <Widget>[
          AspectRatio(
            aspectRatio: _controller.value.aspectRatio,
            child: CameraPreview(_controller),
          ),
          SizedBox(height: 10),
          TextButton(
            child: Text(_isStreaming ? 'Stop Stream' : 'Start Stream'),
            onPressed: _isStreaming ? _stopStream : _startStream,
          ),
        ],
      ),
    );
  }

  void _startStream() async {
    _controller.startImageStream((CameraImage image) {
      // Do something with the image data
      // You can pass the data to your Python script here
    });
    setState(() {
      _isStreaming = true;
    });
  }

  void _stopStream() {
    _controller.stopImageStream();
    setState(() {
      _isStreaming = false;
    });
  }
}
