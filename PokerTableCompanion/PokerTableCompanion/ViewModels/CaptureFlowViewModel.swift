import Foundation
import SwiftUI

@MainActor
final class CaptureFlowViewModel: ObservableObject {
    @Published var draftDetections: [StackDetectionResult] = []
    @Published var observations: [ChipTopObservation] = []
    @Published var isAnalyzing = false
    @Published var hasProcessedImage = false
    @Published var selectedPreviewImage: Image?
    @Published var captureSourceLabel = "No photo selected yet"
    @Published var analysisErrorMessage: String?

    private let analyzer: ChipAnalyzing
    private let sampleLoader = SampleAnalysisLoader()

    init(analyzer: ChipAnalyzing = MockChipAnalyzer()) {
        self.analyzer = analyzer
    }

    func runMockAnalysis(chipConfigs: [ChipColorConfig], sourceLabel: String, previewImage: Image? = nil) async {
        isAnalyzing = true
        hasProcessedImage = false
        analysisErrorMessage = nil
        captureSourceLabel = sourceLabel
        selectedPreviewImage = previewImage
        try? await Task.sleep(nanoseconds: 900_000_000)
        let result = await analyzer.analyze(
            request: ChipAnalysisRequest(chipConfigs: chipConfigs, sourceLabel: sourceLabel)
        )
        observations = result.observations
        draftDetections = result.detections
        captureSourceLabel = result.sourceLabel
        isAnalyzing = false
        hasProcessedImage = true
    }

    func loadBundledSample(chipConfigs: [ChipColorConfig]) async {
        isAnalyzing = true
        hasProcessedImage = false
        analysisErrorMessage = nil
        do {
            let loaded = try sampleLoader.loadSampleCapture(chipConfigs: chipConfigs)
            selectedPreviewImage = loaded.image
            observations = loaded.result.observations
            draftDetections = loaded.result.detections
            captureSourceLabel = loaded.result.sourceLabel
            hasProcessedImage = true
        } catch {
            analysisErrorMessage = "Failed to load sample analysis."
        }
        isAnalyzing = false
    }
}
