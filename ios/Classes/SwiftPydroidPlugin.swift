import Flutter
import UIKit

public class SwiftPydroidPlugin: NSObject, FlutterPlugin {
  public static func register(with registrar: FlutterPluginRegistrar) {
      print("[swift] Registered flutter plugin...")
      let channel = FlutterMethodChannel(name: "pydroid", binaryMessenger: registrar.messenger())
      let instance = SwiftPydroidPlugin()
      registrar.addMethodCallDelegate(instance, channel: channel)
  }

  public func handle(_ call: FlutterMethodCall, result: @escaping FlutterResult) {
      switch call.method {
      case "getPlatformVersion":
          print("[swift] Received method call 'getPlatformVersion'")
          result("\(UIDevice.current.systemName) \(UIDevice.current.systemVersion)")
      case "test":
          print("[swift] test")
          result(100)
      default:
          result(FlutterMethodNotImplemented)
          return
      }
  }
}
