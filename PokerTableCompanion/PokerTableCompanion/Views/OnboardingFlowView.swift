import SwiftUI

struct OnboardingFlowView: View {
    @EnvironmentObject private var appState: AppState
    @Environment(\.dismiss) private var dismiss

    let embeddedInNavigation: Bool

    init(embeddedInNavigation: Bool = false) {
        self.embeddedInNavigation = embeddedInNavigation
    }

    var body: some View {
        content
            .background(AppTheme.background.ignoresSafeArea())
            .navigationTitle(embeddedInNavigation ? "Guided Setup" : "")
            .navigationBarTitleDisplayMode(.inline)
    }

    private var content: some View {
        VStack(spacing: 0) {
            if !embeddedInNavigation {
                header
            }

            TabView {
                setupStep(
                    title: "Teach the capture flow first",
                    body: "This MVP is centered on one job: help a player take a photo, confirm stacks, and see stack value plus BB quickly.",
                    accent: "#1F3B2F",
                    symbol: "viewfinder"
                )

                setupStep(
                    title: "Ask for a clean photo",
                    body: "Tell the player to group same-color stacks together, keep each full stack consistent, and leave loose chips separate.",
                    accent: "#247A57",
                    symbol: "camera.fill"
                )

                setupStep(
                    title: "Set stack size",
                    body: "Before saving the result, confirm how many chips each full stack means. Start with 20 and let the player adjust if needed.",
                    accent: "#A46D1E",
                    symbol: "square.3.layers.3d.down.right"
                )

                setupStep(
                    title: "Map chip colors to value",
                    body: "Each chip color needs a denomination. Once that is set, the review screen can turn stack count and loose chips into total value and BB.",
                    accent: "#C83A3A",
                    symbol: "circle.hexagongrid.fill"
                )
            }
            .tabViewStyle(.page(indexDisplayMode: .always))

            footer
        }
    }

    private var header: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Poker Table Companion")
                .font(.system(size: 34, weight: .bold, design: .rounded))
            Text("Walk through the capture flow once, then test the whole photo-to-summary journey on your iPhone.")
                .foregroundStyle(.white.opacity(0.72))
        }
        .padding(24)
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    private var footer: some View {
        VStack(spacing: 12) {
            NavigationLink {
                SessionSetupView()
            } label: {
                Text("Set blinds")
                    .frame(maxWidth: .infinity)
            }
            .buttonStyle(.bordered)

            NavigationLink {
                ChipSetEditorView()
            } label: {
                Text("Set chip values")
                    .frame(maxWidth: .infinity)
            }
            .buttonStyle(.bordered)

            NavigationLink {
                CaptureIntroView()
            } label: {
                Text("Start capture flow")
                    .frame(maxWidth: .infinity)
            }
            .buttonStyle(.borderedProminent)

            if !embeddedInNavigation {
                Button("Enter app") {
                    appState.completeOnboarding()
                }
                .buttonStyle(.plain)
                .padding(.top, 4)
            } else {
                Button("Done") {
                    dismiss()
                }
                .buttonStyle(.plain)
                .padding(.top, 4)
            }
        }
        .padding(24)
    }

    private func setupStep(title: String, body: String, accent: String, symbol: String) -> some View {
        VStack(alignment: .leading, spacing: 20) {
            RoundedRectangle(cornerRadius: 30, style: .continuous)
                .fill(
                    LinearGradient(
                        colors: [Color(hex: accent), Color.white.opacity(0.25)],
                        startPoint: .topLeading,
                        endPoint: .bottomTrailing
                    )
                )
                .frame(height: 260)
                .overlay(
                    Image(systemName: symbol)
                        .font(.system(size: 68, weight: .medium))
                        .foregroundStyle(.white)
                )

            Text(title)
                .font(.system(size: 30, weight: .bold, design: .rounded))
            Text(body)
                .font(.body)
                .foregroundStyle(.white.opacity(0.74))

            Spacer()
        }
        .padding(24)
    }
}

#Preview {
    NavigationStack {
        OnboardingFlowView()
            .environmentObject(AppState())
    }
}
