import 'dart:ui' as ui;

import 'package:flutter/material.dart';

class FacePainter extends CustomPainter {
  /// draws the bounds of the face & forehead on a specified canvas
  /// Canvas extends the full length of its container, it's the Image widget that is cropped

  BuildContext ctx;
  Rect face; // bounding rectangle of face
  Rect forehead;
  ui.Image? image;
  FacePainter(this.ctx, this.face, this.forehead);

  @override
  void paint(ui.Canvas canvas, ui.Size size) {
    // if (image != null) {
    //   // ***hack
    //   log("image exists!");
    //   // image is loaded async
    //   // try this instead of displaying image file...
    //   paintImage(
    //       canvas: canvas,
    //       rect: Rect.fromPoints(
    //           const Offset(0, 0), Offset(size.width, size.height)),
    //       image: image!);
    //   // return;
    // } else {
    //   log("image is NULL!");
    // }

    const double strokeWidth = 2.0;
    final Paint referencePaint = Paint()
      ..style = PaintingStyle.stroke
      ..strokeWidth = strokeWidth
      ..color = Colors.white;
    final Paint headPaint = Paint()
      ..style = PaintingStyle.stroke
      ..strokeWidth = strokeWidth * 2
      ..color = Colors.lime.shade300;
    final Paint foreheadPaint = Paint()
      ..style = PaintingStyle.stroke
      ..strokeWidth = strokeWidth / 2
      ..color = Colors.lime.shade200;

    final double screenWidth = size.width;
    final double screenHeight = size.height;
    // log("Painter Screen: W = $screenWidth H = $screenHeight");
    var boxSize = 150.0; // height/width of ref box

    final topLeftBox = Rect.fromCenter(
        center: Offset(boxSize / 2, boxSize / 2),
        width: boxSize,
        height: boxSize);
    final midBox = Rect.fromCenter(
        center: Offset(screenWidth / 2, screenHeight / 2),
        width: boxSize,
        height: boxSize);
    final bottomRightBox = Rect.fromCenter(
        center: Offset(screenWidth - boxSize / 2, screenHeight - boxSize / 2),
        width: boxSize,
        height: boxSize);

    // draw reference boxes & face/forehead
    canvas.drawLine(Offset(screenWidth / 2, 0),
        Offset(screenWidth / 2, screenHeight), referencePaint);
    canvas.drawLine(Offset(0, screenHeight / 2),
        Offset(screenWidth, screenHeight / 2), referencePaint);
    canvas.drawRect(topLeftBox, referencePaint);
    canvas.drawRect(midBox, referencePaint);
    canvas.drawRect(bottomRightBox, referencePaint);

    double xOffset = -25;
    double yOffset = -75;
    // var shiftedBox = Rect.fromPoints(
    //     _shiftPoint(face.topLeft, xOffset, yOffset),
    //     _shiftPoint(face.bottomRight, xOffset, yOffset));
    canvas.drawRect(face, headPaint);
    // canvas.drawRect(shiftedBox, foreheadPaint);
    canvas.drawRect(forehead, foreheadPaint);
  }

  Offset _shiftPoint(Offset point, double byX, double byY) {
    return Offset(point.dx + byX, point.dy + byY);
  }

  @override
  bool shouldRepaint(FacePainter oldDelegate) {
    return face != oldDelegate.face || forehead != oldDelegate.forehead;
  }
}
