import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import 'package:pydroid/pydroid.dart';
import 'dart:io';
import 'package:pydroid_example/face_detection/canvas.dart';
import 'package:image/image.dart' as imglib;

class CameraScreen extends StatefulWidget {
  const CameraScreen({Key? key}) : super(key: key);

  @override
  _CameraScreenState createState() => _CameraScreenState();
}

class _CameraScreenState extends State<CameraScreen> {
  late CameraController _controller;
  late List<CameraDescription> _cameras;

  bool _streamingBegun = false;
  bool _isProcessing = false;

  Rect _face = Rect.zero;
  final Rect _forehead = Rect.zero;
  String _path = '';
  String tracker_path = 'tracker.sav';
  String hardcodedPath =
      '/data/user/0/com.jacobsonlab.pydroid_example/app_flutter/Pictures/flutter_test';

  @override
  void initState() {
    super.initState();
    _initializeCamera();
  }

  Future<void> _initializeCamera() async {
    _cameras = await availableCameras();
    _controller = CameraController(_cameras.last, ResolutionPreset.medium);

    _controller.initialize().then((_) {
      if (!mounted) return;
      setState(() {});
    });
  }

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    if (!_controller.value.isInitialized) {
      return Container();
    }
    return Scaffold(
      appBar: AppBar(
        title: const Text("Camera"),
      ),
      body: Column(
        children: <Widget>[
          SizedBox(
            width: 480,
            height: 300,
            child: AspectRatio(
              child: CustomPaint(
                foregroundPainter: FacePainter(
                  context,
                  _face,
                  _forehead,
                ),
                child: CameraPreview(_controller),
              ),
              aspectRatio: _controller.value.aspectRatio,
            ),
          ),
          Container(
            child: CustomPaint(
              foregroundPainter: FacePainter(
                context,
                _face,
                _forehead,
              ), // get reference to facePainter so we can update the image object ***hack,
              child: Image.file(
                File(_path),
                scale: 0.75,
              ),
              // child: const SizedBox.shrink(),
            ),
          ),
          const SizedBox(height: 10),
          TextButton(
            child: Text(_streamingBegun ? 'Stop Stream' : 'Start Stream'),
            onPressed: _streamingBegun ? _stopStream : _startStream,
          ),
        ],
      ),
    );
  }

  Future<String> saveImageFile(imageBytes, count) async {
    await Directory(hardcodedPath).create(recursive: true);
    final String filePath = '$hardcodedPath/$count.png';

    // if a capture is already pending, do nothing
    if (_controller.value.isTakingPicture) return '';

    try {
      File file = File(filePath);
      file.writeAsBytes(imageBytes);
      return filePath;
    } on CameraException {
      return '';
    }
  }

  Future<List<int>> convertImagetoPng(CameraImage image) async {
    try {
      late imglib.Image img;
      if (image.format.group == ImageFormatGroup.yuv420) {
        img = _convertYUV420(image);
      } else if (image.format.group == ImageFormatGroup.bgra8888) {
        img = _convertBGRA8888(image);
      } else {
        print('we ended up in here');
      }

      imglib.PngEncoder pngEncoder = imglib.PngEncoder();

      // Convert to png
      List<int> png = pngEncoder.encode(img);
      return png;
    } catch (e) {
      print(">>>>>>>>>>>> ERROR:" + e.toString());
      return [];
    }
  }

  // CameraImage BGRA8888 -> PNG
  // Color
  imglib.Image _convertBGRA8888(CameraImage image) {
    return imglib.Image.fromBytes(
      width: image.width,
      height: image.height,
      bytes: image.planes[0].bytes.buffer,
      // format: imglib.Format.bgra,
    );
  }

  // CameraImage YUV420_888 -> PNG -> Image (compresion:0, filter: none)
  // Black
  imglib.Image _convertYUV420(CameraImage image) {
    var img = imglib.Image(
        width: image.width, height: image.height); // Create Image buffer

    Plane plane = image.planes[0];
    const int shift = (0xFF << 24);

    // Fill image buffer with plane[0] from YUV420_888
    for (int x = 0; x < image.width; x++) {
      for (int planeOffset = 0;
          planeOffset < image.height * image.width;
          planeOffset += image.width) {
        final pixelColor = plane.bytes[planeOffset + x];
        // color: 0x FF  FF  FF  FF
        //           A   B   G   R
        // Calculate pixel color
        var newVal =
            shift | (pixelColor << 16) | (pixelColor << 8) | pixelColor;

        if (img.data != null) {
          img.data!.toUint8List()[planeOffset + x] = newVal;
        }
      }
    }

    return img;
  }

  void _startStream() async {
    // for having unique file names
    var count = 0;

    // start streaming images
    _controller.startImageStream((CameraImage image) async {
      List<int> imageBytes = await convertImagetoPng(image);

      // call save image file method
      saveImageFile(imageBytes, count).then((res) async {
        final pathToPass = _streamingBegun ? tracker_path : '';
        setState(() {
          _isProcessing = true;
        });
        Pydroid.analyzeStream(res, pathToPass).then((value) {
          setState(() {
            _face = value;
            _path = res;
          });
        });
      });
    }).catchError((err) => {print("error on save image file error: $err")});

    count += 1;

    setState(() {
      _streamingBegun = true;
    });
  }

  void _stopStream() {
    _controller.stopImageStream();
    setState(() {
      _streamingBegun = false;
    });
  }
}
