import SwiftUI

struct RootView: View {
    @EnvironmentObject private var appState: AppState

    var body: some View {
        NavigationStack {
            DashboardView()
        }
        .tint(AppTheme.chipAccent)
        .fullScreenCover(
            isPresented: Binding(
                get: { !appState.hasCompletedOnboarding },
                set: { isPresented in
                    if !isPresented {
                        appState.completeOnboarding()
                    }
                }
            )
        ) {
            NavigationStack {
                OnboardingFlowView()
                    .environmentObject(appState)
            }
        }
    }
}

#Preview {
    RootView()
        .environmentObject(AppState())
}
