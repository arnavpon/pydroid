import 'package:flutter/services.dart';
import 'package:flutter_test/flutter_test.dart';

import 'package:pydroid/pydroid.dart';

void main() {
  const MethodChannel channel = MethodChannel('pydroid');

  TestWidgetsFlutterBinding.ensureInitialized();

  setUp(() {
    handler(MethodCall methodCall) async {
      if (methodCall.method == "getPlatformVersion") {
        return 0;
      }
      return null;
    }

    TestWidgetsFlutterBinding.ensureInitialized();
    TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
        .setMockMethodCallHandler(channel, handler);
  });

  tearDown(() {
    channel.setMockMethodCallHandler(null);
  });

  test('getPlatformVersion', () async {
    // mock this...
    expect(await Pydroid.platformVersion, '42');
  });

  test('getBatteryLevel', () async {
    // mock this to check the python version, which should be 3.8.11
    expect(await Pydroid.runTest(), '-1');
  });
}
