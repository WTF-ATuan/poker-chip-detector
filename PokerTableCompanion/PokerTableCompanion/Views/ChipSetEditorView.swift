import SwiftUI

struct ChipSetEditorView: View {
    @EnvironmentObject private var appState: AppState

    var body: some View {
        List {
            Section("Chip Colors") {
                ForEach($appState.chipConfigs) { $chip in
                    VStack(alignment: .leading, spacing: 12) {
                        HStack {
                            Circle()
                                .fill(chip.swiftUIColor)
                                .frame(width: 18, height: 18)
                            TextField("Color name", text: $chip.name)
                        }
                        Stepper("Denomination: \(chip.denomination.formattedCurrency)", value: $chip.denomination, in: 25...100000, step: 25)
                        Stepper("Stack size: \(chip.stackSize)", value: $chip.stackSize, in: 5...40, step: 5)
                    }
                    .padding(.vertical, 8)
                }
            }

            Section {
                Text("MVP assumes each full stack is a fixed count. That makes photo-assisted counting much easier than estimating stack height.")
                    .foregroundStyle(.secondary)
            }
        }
        .navigationTitle("Chip Set")
        .navigationBarTitleDisplayMode(.inline)
    }
}

#Preview {
    NavigationStack {
        ChipSetEditorView()
            .environmentObject(AppState())
    }
}
