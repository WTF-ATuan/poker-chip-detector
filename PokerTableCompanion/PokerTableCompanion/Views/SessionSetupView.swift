import SwiftUI

struct SessionSetupView: View {
    @EnvironmentObject private var appState: AppState

    var body: some View {
        Form {
            Section("Session") {
                TextField("Session name", text: $appState.sessionName)
                Stepper("Small blind: \(appState.smallBlind)", value: $appState.smallBlind, in: 25...5000, step: 25)
                Stepper("Big blind: \(appState.bigBlind)", value: $appState.bigBlind, in: 50...10000, step: 50)
            }

            Section("Why this matters") {
                Text("The app uses your BB to turn any future chip estimate into a quick table-ready number.")
                    .foregroundStyle(.secondary)
            }
        }
        .navigationTitle("Session Setup")
        .navigationBarTitleDisplayMode(.inline)
    }
}

#Preview {
    NavigationStack {
        SessionSetupView()
            .environmentObject(AppState())
    }
}
