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

  HRScreen({Key? key, required this.hr})
      : userHr = hr[0].round(),
        userHRV = hr[1] as double, super(key: key);

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('HR Screen')),
      body: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Text('User HR: $userHr', style: const TextStyle(fontSize: 24)),
            const SizedBox(height: 20),
            Text('User HRV: ${userHRV.toStringAsFixed(2)}', style: const TextStyle(fontSize: 24)),
          ],
        ),
      ),
    );
  }
}

