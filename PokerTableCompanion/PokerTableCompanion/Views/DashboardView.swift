import SwiftUI

struct DashboardView: View {
    @EnvironmentObject private var appState: AppState

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 20) {
                heroCard
                summaryGrid
                quickActions
                workflowCard
                chipConfigCard
            }
            .padding(20)
        }
        .background(AppTheme.background.ignoresSafeArea())
        .navigationTitle("Poker Table")
    }

    private var heroCard: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text(appState.sessionName)
                .font(.system(size: 30, weight: .bold, design: .rounded))
            Text("Guide the player from photo capture to stack value and BB in one short flow.")
                .foregroundStyle(.white.opacity(0.78))
            HStack {
                Label("SB \(appState.smallBlind)", systemImage: "circle.lefthalf.filled")
                Spacer()
                Label("BB \(appState.bigBlind)", systemImage: "circle.fill")
            }
            .font(.headline)
        }
        .padding(20)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(
            LinearGradient(
                colors: [Color(hex: "#1F3B2F"), Color(hex: "#315C48")],
                startPoint: .topLeading,
                endPoint: .bottomTrailing
            )
        )
        .foregroundStyle(.white)
        .clipShape(RoundedRectangle(cornerRadius: 28, style: .continuous))
    }

    private var summaryGrid: some View {
        VStack(spacing: 12) {
            SummaryCard(title: "Current Stack", value: appState.sessionSummary.totalChipsValue.formattedCurrency)
            SummaryCard(title: "Current BB", value: appState.sessionSummary.bigBlinds.bbText)
            SummaryCard(
                title: "Last Capture",
                value: appState.lastCaptureDate.map { $0.formatted(date: .abbreviated, time: .shortened) } ?? "Not captured"
            )
        }
    }

    private var quickActions: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Main Flow")
                .font(.title3.weight(.semibold))

            NavigationLink {
                SessionSetupView()
            } label: {
                ActionRow(title: "Set blinds and session", subtitle: "Prepare the table settings", color: "#C83A3A")
            }

            NavigationLink {
                ChipSetEditorView()
            } label: {
                ActionRow(title: "Configure chip colors", subtitle: "Map colors to denominations", color: "#A46D1E")
            }

            if AppFeatures.cameraCaptureEnabled {
                NavigationLink {
                    CaptureIntroView()
                } label: {
                    ActionRow(title: "Run capture flow", subtitle: "Photo, review, and BB summary", color: "#247A57")
                }

                NavigationLink {
                    OnboardingFlowView(embeddedInNavigation: true)
                } label: {
                    ActionRow(title: "View guided setup", subtitle: "Teach players how to capture correctly", color: "#1F3B2F")
                }
            }
        }
    }

    private var workflowCard: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Capture Workflow")
                .font(.title3.weight(.semibold))
            workflowRow(number: "1", title: "Frame the chips", subtitle: "Ask the player to group same-color stacks together.")
            workflowRow(number: "2", title: "Confirm stack size", subtitle: "Set how many chips a full stack represents.")
            workflowRow(number: "3", title: "Map color values", subtitle: "Check each color denomination before saving.")
            workflowRow(number: "4", title: "Review estimate", subtitle: "Adjust stack count and loose chips, then save.")
        }
        .cardStyle()
    }

    private var chipConfigCard: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Chip Set")
                .font(.title3.weight(.semibold))
            ForEach(appState.chipConfigs) { chip in
                HStack {
                    Circle()
                        .fill(chip.swiftUIColor)
                        .frame(width: 18, height: 18)
                        .overlay(Circle().stroke(Color.black.opacity(0.12), lineWidth: 1))
                    Text(chip.name)
                    Spacer()
                    Text("\(chip.denomination.formattedCurrency) / stack \(chip.stackSize)")
                        .foregroundStyle(.secondary)
                        .font(.subheadline)
                }
            }
        }
        .cardStyle()
    }

    private func workflowRow(number: String, title: String, subtitle: String) -> some View {
        HStack(alignment: .top, spacing: 12) {
            Text(number)
                .font(.headline)
                .foregroundStyle(.white)
                .frame(width: 28, height: 28)
                .background(Color(hex: "#1F3B2F"))
                .clipShape(Circle())
            VStack(alignment: .leading, spacing: 4) {
                Text(title)
                    .font(.headline)
                Text(subtitle)
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
            }
        }
    }
}

private struct SummaryCard: View {
    let title: String
    let value: String

    var body: some View {
        HStack {
            Text(title)
                .foregroundStyle(.white.opacity(0.7))
            Spacer()
            Text(value)
                .font(.title3.weight(.bold))
        }
        .padding(18)
        .frame(maxWidth: .infinity)
        .background(AppTheme.cardAlt)
        .overlay(
            RoundedRectangle(cornerRadius: 22, style: .continuous)
                .stroke(AppTheme.stroke, lineWidth: 1)
        )
        .clipShape(RoundedRectangle(cornerRadius: 22, style: .continuous))
    }
}

private struct ActionRow: View {
    let title: String
    let subtitle: String
    let color: String

    var body: some View {
        HStack(spacing: 14) {
            RoundedRectangle(cornerRadius: 16, style: .continuous)
                .fill(Color(hex: color))
                .frame(width: 44, height: 44)
                .overlay(Image(systemName: "arrow.up.right").foregroundStyle(.white))
            VStack(alignment: .leading, spacing: 4) {
                Text(title)
                    .font(.headline)
                    .foregroundStyle(.white)
                Text(subtitle)
                    .font(.subheadline)
                    .foregroundStyle(.white.opacity(0.7))
            }
            Spacer()
        }
        .padding(18)
        .background(AppTheme.card)
        .overlay(
            RoundedRectangle(cornerRadius: 22, style: .continuous)
                .stroke(AppTheme.stroke, lineWidth: 1)
        )
        .clipShape(RoundedRectangle(cornerRadius: 22, style: .continuous))
    }
}

#Preview {
    NavigationStack {
        DashboardView()
            .environmentObject(AppState())
    }
}
