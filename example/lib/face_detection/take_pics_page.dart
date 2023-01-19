import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:camera/src/camera_controller.dart';
// import 'package:path_provider/path_provider.dart';
import 'package:pydroid/pydroid.dart';
import 'dart:io';
import 'package:pydroid_example/face_detection/canvas.dart';
import 'package:image/image.dart' as imglib;

class CameraScreen extends StatefulWidget {
  @override
  _CameraScreenState createState() => _CameraScreenState();
}

class _CameraScreenState extends State<CameraScreen> {
  late CameraController _controller;
  late List<CameraDescription> _cameras;
  bool _isStreaming = false;

  Rect _face = Rect.zero;
  Rect _forehead = Rect.zero;
  bool _started = false;
  String _path = '';
  String tracker_path = 'tracker.sav';
  String hardcodedPath = '/data/user/0/com.jacobsonlab.pydroid_example/app_flutter/Pictures/flutter_test';

  @override
  void initState() {
    super.initState();
    _initializeCamera();
  }

  Future<void> _initializeCamera() async {
    _cameras = await availableCameras();
    _controller = CameraController(_cameras.last, ResolutionPreset.medium);//, imageFormatGroup: ImageFormatGroup.jpeg);

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
          SizedBox(height: 10),
          TextButton(
            child: Text(_isStreaming ? 'Stop Stream' : 'Start Stream'),
            onPressed: _isStreaming ? _stopStream : _startStream,
          ),
        ],
      ),
    );
  }

  Future<String> saveImageFile(imageBytes, count) async {
    // final Directory extDir = await getApplicationDocumentsDirectory();
    // final String dirPath = '${extDir.path}/Pictures/flutter_test';
    // print('DDIDDIIDD');
    // print(dirPath);
    await Directory(hardcodedPath).create(recursive: true);
    final String filePath = '${hardcodedPath}/${count}.png';

    if (_controller.value.isTakingPicture) {
      // A capture is already pending, do nothing.
      return '';
    }

    try {
      File file = new File(filePath);
      file.writeAsBytes(imageBytes);
      // print("finish image saved ${imageBytes}");
    } on CameraException catch (e) {
      // _showCameraException(e);
      return '';
    }
    return filePath;
  }

  Future<List<int>> convertImagetoPng(CameraImage image) async {
    try {
      late imglib.Image img;
      print('the group is');
      print(image.format.group);
      if (image.format.group == ImageFormatGroup.yuv420) {
        img = _convertYUV420(image);
      } else if (image.format.group == ImageFormatGroup.bgra8888) {
        img = _convertBGRA8888(image);
      } else {
        print('we ended up in here');
      }

      imglib.PngEncoder pngEncoder = new imglib.PngEncoder();

      // Convert to png
      List<int> png = pngEncoder.encodeImage(img);
      return png;
    } catch (e) {
      print(">>>>>>>>>>>> ERROR:" + e.toString());
    }
    return [];
  }

  // CameraImage BGRA8888 -> PNG
  // Color
  imglib.Image _convertBGRA8888(CameraImage image) {
    return imglib.Image.fromBytes(
      image.width,
      image.height,
      image.planes[0].bytes,
      format: imglib.Format.bgra,
    );
  }

  // CameraImage YUV420_888 -> PNG -> Image (compresion:0, filter: none)
  // Black
  imglib.Image _convertYUV420(CameraImage image) {
    var img = imglib.Image(image.width, image.height); // Create Image buffer

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
        var newVal = shift | (pixelColor << 16) | (pixelColor << 8) | pixelColor;

        img.data[planeOffset + x] = newVal;
      }
    }

    return img;
  }

  void _startStream() async {
    var count = 0;
    _controller.startImageStream((CameraImage image) async {
      print('we start with');
      print(image.format.group);
      // print("img format: ${image.format} planes: ${image.planes}");
      // List<int> imageBytes = [];

      // for (var i = image.planes.length - 1; i >= 0; i--) {
      //   var plane = image.planes[i];
      //   imageBytes.addAll(plane.bytes.toList());
      // }
      // // image.planes.map((plane) {
      // //   print('We have bytes here: ${plane.bytes.toList()}');
      // //   imageBytes.addAll(plane.bytes.toList());
      // // });
      // print('got planes');

      List<int> imageBytes = await convertImagetoPng(image);
      print('the bytes');
      print(imageBytes);
      
      // call save image file method
      saveImageFile(imageBytes, count).then((res) async {
        print("save image file successfull filepath: $res");

        // print('loading test');
        // var test = Image.file(File(res));
        // print('TEST: ${test}');

        print("[STREAM] Analyzing...");
        if (!_started) {
          Pydroid.analyzeStream(res, '').then((value) {
            // var totalTime = DateTime.now().difference(startTime).inMilliseconds;
            // log("[Flutter] Received value '$value' in $totalTime milliseconds");
            setState(() {
              _face = value;
              _path = res;
              print('Set face to:');
              print(value);
              // facePainter!.face = value; // ***for hack
              // _isComputationRunning = false; // unset blocker
            });
          });
        } else {
          Pydroid.analyzeStream(res, tracker_path).then((value) {
            // var totalTime = DateTime.now().difference(startTime).inMilliseconds;
            // log("[Flutter] Received value '$value' in $totalTime milliseconds");
            setState(() {
              _face = value;
              _path = res;
              print('Set face to:');
              print(value);
              // facePainter!.face = value; // ***for hack
              // _isComputationRunning = false; // unset blocker
            });
          });
        }

      }).catchError((err) => {
        print("error on save image file error: $err")
      });

      count += 1;

      // Save the image to a file
      // final directory = await getApplicationDocumentsDirectory();
      // final path = '${directory.path}/image${counter}.jpg';
      // final file = File(path);
      // final bytes = image.planes.first.bytes;
      // await file.writeAsBytes(bytes);
      // counter += 1;
      
      // Pass the file path to the Python script
      // print("[STREAM] Analyzing...");
      // await Pydroid.analyzeStream(path);
    });
    setState(() {
      _isStreaming = true;;
    });
  }

  void _stopStream() {
    _controller.stopImageStream();
    setState(() {
      _isStreaming = false;
    });
  }
}
