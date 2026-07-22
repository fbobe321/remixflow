import SwiftUI

// The app entry point. Add these App/*.swift files to an Xcode iOS App target and
// add the RemixFlowKit package (see ios/README.md).
@main
struct RemixFlowApp: App {
    var body: some Scene {
        WindowGroup {
            ContentView()
                .preferredColorScheme(.dark)
        }
    }
}
