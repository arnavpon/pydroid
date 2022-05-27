import 'dart:ui' as ui;
import 'dart:math';
import 'package:flutter/material.dart';

// subset rnd part image, red/green val

class FacePainter extends CustomPainter {
  /// draws the bounds of the face & forehead on a specified canvas?

  BuildContext ctx;
  final Rect face; // bounding rectangle of face
  // Rect get faceBounds => (face == null) ? Rect.zero : face.boundingBox;
  Rect forehead;
  FacePainter(this.ctx, this.face, this.forehead);

  @override
  void paint(ui.Canvas canvas, ui.Size size) {
    final double strokeWidth = 2.0;
    final Paint referencePaint = Paint()
      ..style = PaintingStyle.stroke
      ..strokeWidth = strokeWidth
      ..color = Colors.white;
    final Paint headPaint = Paint()
      ..style = PaintingStyle.stroke
      ..strokeWidth = strokeWidth
      ..color = Colors.lime.shade300;
    final Paint foreheadPaint = Paint()
      ..style = PaintingStyle.stroke
      ..strokeWidth = strokeWidth / 2
      ..color = Colors.lime.shade200;

    final double screenWidth = 300;
    final double screenHeight = 600;
    print("Painter Screen: W = $screenWidth H = $screenHeight");
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
    canvas.drawRect(face, headPaint);
    canvas.drawRect(forehead, foreheadPaint);

    // rotations & stuff might be to get canvas & image to match each other...

    // canvas.rotate(
    //     -pi / 2); // if you don't rotate, box doesn't show up on canvas, why?
    // canvas.translate(-(MediaQuery.of(context).size.height * 0.77) ?? 0,
    //     MediaQuery.of(context).size.width * 1.15); // ?
    // canvas.scale(1.9, -1.9); // ?
  }

  @override
  bool shouldRepaint(FacePainter oldDelegate) {
    return face != oldDelegate.face || forehead != oldDelegate.forehead;
  }
}
