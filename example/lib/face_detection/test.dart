import 'dart:ffi';
import 'dart:io';
import 'dart:typed_data';

import 'package:ffi/ffi.dart';
import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import 'package:image/image.dart' as imglib;

typedef convert_func = Pointer<Uint32> Function(
   Pointer<Uint8>, Pointer<Uint8>, Pointer<Uint8>,
   Int32, Int32, Int32, Int32
);
typedef Convert = Pointer<Uint32> Function(
   Pointer<Uint8>, Pointer<Uint8>, Pointer<Uint8>,
   int, int, int, int
);

class MyHomePage extends StatefulWidget {
  MyHomePage({Key? key, required this.title}) : super();
  final String title;
  @override
  _MyHomePageState createState() => _MyHomePageState();
}

class _MyHomePageState extends State<MyHomePage> {

  late CameraController _camera;
  bool _cameraInitialized = false;
  late CameraImage _savedImage;

  final DynamicLibrary convertImageLib = Platform.isAndroid
    ? DynamicLibrary.open("libconvertImage.so")
    : DynamicLibrary.process();
  late Convert conv;


  @override
  void initState(){
    super.initState();
    _initializeCamera();
    conv = convertImageLib
    .lookup<NativeFunction<convert_func>>('convertImage')
    .asFunction<Convert>();
  }
  
  void _initializeCamera() async {
    // Get list of cameras of the device
    List<CameraDescription> cameras = await availableCameras();
  // Create the CameraController
    _camera = new CameraController(
      cameras[0], ResolutionPreset.veryHigh
    );
  // Initialize the CameraController
    _camera.initialize().then((_) async{
      // Start ImageStream
      await _camera.startImageStream((CameraImage image) =>
        _processCameraImage(image));
        setState(() {
          _cameraInitialized = true;
        });
    });
  }
  void _processCameraImage(CameraImage image) async {
    setState(() {
      _savedImage = image;
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text(widget.title),
      ),
      body: Center(
        child:
          (_cameraInitialized)
          ? AspectRatio(aspectRatio: _camera.value.aspectRatio,
              child: CameraPreview(_camera),)
          : CircularProgressIndicator()
      ),
      floatingActionButton: FloatingActionButton(
        onPressed: (){
          // Allocate memory for the 3 planes of the image
          // Pointer<Uint8> p = allocate(count: _savedImage.planes[0].bytes.length);
          Pointer<Uint8> p = calloc<Uint8>();
          Pointer<Uint8> p1 = calloc<Uint8>();
          Pointer<Uint8> p2 = calloc<Uint8>();
          p.value = _savedImage.planes[0].bytes.length;
          p1.value = _savedImage.planes[1].bytes.length;
          p2.value = _savedImage.planes[2].bytes.length;
          // Pointer<Uint8> p1 = allocate(count: _savedImage.planes[1].bytes.length);
          // Pointer<Uint8> p2 = allocate(count: _savedImage.planes[2].bytes.length);
          
          // Assign the planes data to the pointers of the image
          Uint8List pointerList = p.asTypedList(_savedImage.planes[0].bytes.length);
          Uint8List pointerList1 = p1.asTypedList(_savedImage.planes[1].bytes.length);
          Uint8List pointerList2 = p2.asTypedList(_savedImage.planes[2].bytes.length);
          pointerList.setRange(0, _savedImage.planes[0].bytes.length, _savedImage.planes[0].bytes);
          pointerList1.setRange(0, _savedImage.planes[1].bytes.length, _savedImage.planes[1].bytes);
          pointerList2.setRange(0, _savedImage.planes[2].bytes.length, _savedImage.planes[2].bytes);
          print('this point');
          // Call the convertImage function and convert the YUV to RGB
          Pointer<Uint32> imgP = conv(p, p1, p2, _savedImage.planes[1].bytesPerRow,
            _savedImage.planes[1]?.bytesPerPixel ?? 0, _savedImage.width, _savedImage.height);
          // Get the pointer of the data returned from the function to a List
          List<int> imgData = imgP.asTypedList((_savedImage.width * _savedImage.height));
          print('got here sam');
          // Generate image from the converted data  
          imglib.Image img = imglib.Image.fromBytes(_savedImage.height, _savedImage.width, imgData);
          
          // Free the memory space allocated
          // from the planes and the converted data
          print('at frees');
          calloc.free(p);
          calloc.free(p1);
          calloc.free(p2);
          calloc.free(imgP);
          print('survived frees');
        },
        tooltip: 'Increment',
        child: Icon(Icons.camera_alt),
      ), // This trailing comma makes auto-formatting nicer for build methods.
      floatingActionButtonLocation: FloatingActionButtonLocation.centerFloat,
    );
  }
}