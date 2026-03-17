import Foundation

final class AppState: ObservableObject {
    @Published var hasCompletedOnboarding = false
    @Published var sessionName = "Friday Cash Game"
    @Published var smallBlind = 100
    @Published var bigBlind = 200
    @Published var chipConfigs: [ChipColorConfig] = [
        ChipColorConfig(name: "Orange", denomination: 5000, colorHex: "#E05F34"),
        ChipColorConfig(name: "Pink", denomination: 1000, colorHex: "#D98AA8"),
        ChipColorConfig(name: "Green", denomination: 500, colorHex: "#3C8D68"),
        ChipColorConfig(name: "Black", denomination: 100, colorHex: "#262626")
    ]
    @Published var latestDetections: [StackDetectionResult] = []
    @Published var lastCaptureDate: Date?

    var totalChipValue: Int {
        latestDetections.reduce(0) { partial, detection in
            guard let config = chipConfigs.first(where: { $0.id == detection.chipColorID }) else {
                return partial
            }
            let count = detection.stackCount * config.stackSize + detection.looseCount
            return partial + count * config.denomination
        }
    }

    var currentBigBlinds: Double {
        guard bigBlind > 0 else { return 0 }
        return Double(totalChipValue) / Double(bigBlind)
    }

    var sessionSummary: SessionSummary {
        SessionSummary(
            totalChipsValue: totalChipValue,
            bigBlinds: currentBigBlinds
        )
    }

    func applyDetections(_ detections: [StackDetectionResult]) {
        latestDetections = detections
        lastCaptureDate = .now
    }

    func completeOnboarding() {
        hasCompletedOnboarding = true
    }
}
