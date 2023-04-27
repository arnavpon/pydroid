// import 'package:flutter/material.dart';

// class HRScreen extends StatelessWidget {
//   final dynamic hr;

//   HRScreen({required this.hr});

//   @override
//   Widget build(BuildContext context) {
//     return Scaffold(
//       appBar: AppBar(title: Text('HR Screen')),
//       body: Center(child: Text('HR value: $hr')),
//     );
//   }
// }
import 'package:flutter/material.dart';

class HRScreen extends StatelessWidget {
  final List<dynamic> hr;
  final int userHr;
  final double userHRV;

  HRScreen({required this.hr})
      : userHr = hr[0].round(),
        userHRV = hr[1] as double;

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: Text('HR Screen')),
      body: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Text('User HR: $userHr', style: TextStyle(fontSize: 24)),
            SizedBox(height: 20),
            Text('User HRV: ${userHRV.toStringAsFixed(2)}', style: TextStyle(fontSize: 24)),
          ],
        ),
      ),
    );
  }
}

