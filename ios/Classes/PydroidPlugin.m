#import "PydroidPlugin.h"
#if __has_include(<pydroid/pydroid-Swift.h>)
#import <pydroid/pydroid-Swift.h>
#else
// Support project import fallback if the generated compatibility header
// is not copied when this plugin is created as a library.
// https://forums.swift.org/t/swift-static-libraries-dont-copy-generated-objective-c-header/19816
#import "pydroid-Swift.h"
#endif

@implementation PydroidPlugin
+ (void)registerWithRegistrar:(NSObject<FlutterPluginRegistrar>*)registrar {
  [SwiftPydroidPlugin registerWithRegistrar:registrar];
}
@end
