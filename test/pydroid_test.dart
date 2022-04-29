import 'package:flutter/services.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:pydroid/pydroid.dart';

void main() {
  const MethodChannel channel = MethodChannel('pydroid');

  TestWidgetsFlutterBinding.ensureInitialized();

  setUp(() {
    // channel.setMockMethodCallHandler((MethodCall methodCall) async {
    //   return '42';
    // });
  });

  tearDown(() {
    channel.setMockMethodCallHandler(null);
  });

  test('getPlatformVersion', () async {
    expect(await Pydroid.platformVersion, '42');
  });

  test('getBatteryLevel', () async {
    expect(await Pydroid.batteryLevel, '-1');
  });
}
