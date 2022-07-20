import 'package:image_picker/image_picker.dart';
import 'package:flutter/material.dart';

import 'display_picture_screen.dart';

class ChoosePictureScreen extends StatefulWidget {
  const ChoosePictureScreen({Key? key}) : super(key: key);

  @override
  State<ChoosePictureScreen> createState() => _ChoosePictureScreenState();
}

class _ChoosePictureScreenState extends State<ChoosePictureScreen> {
  final ImagePicker _picker = ImagePicker();

  Future selectImage() async {
    final XFile? image = await _picker.pickImage(source: ImageSource.gallery);
    if (image != null) {
      await Navigator.of(context).push(
        MaterialPageRoute(
          builder: (context) => DisplayPictureScreen(
            image: image,
          ),
        ),
      );
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text("Choose Picture")),
      body: SizedBox.expand(
        child: FloatingActionButton(
          child: const Text("Select Image"),
          onPressed: selectImage,
        ),
      ),
    );
  }
}
