import SwiftUI

struct RootView: View {
    var body: some View {
        NavigationStack {
            HandRecorderView()
        }
        .tint(AppTheme.chipAccent)
    }
}

#Preview {
    RootView()
        .environmentObject(AppState())
}
