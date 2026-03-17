import SwiftUI

struct RootView: View {
    var body: some View {
        NavigationStack {
            CaptureIntroView()
        }
        .tint(AppTheme.chipAccent)
    }
}

#Preview {
    RootView()
        .environmentObject(AppState())
}
