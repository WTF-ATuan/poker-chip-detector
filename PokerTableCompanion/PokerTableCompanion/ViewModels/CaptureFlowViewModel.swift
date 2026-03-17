import Foundation
import SwiftUI
import UIKit

@MainActor
final class CaptureFlowViewModel: ObservableObject {
    @Published var draftDetections: [StackDetectionResult] = []
    @Published var observations: [ChipTopObservation] = []
    @Published var isAnalyzing = false
    @Published var hasProcessedImage = false
    @Published var selectedPreviewImage: Image?
    @Published var selectedPreviewImageSize: CGSize?
    @Published var captureSourceLabel = "No photo selected yet"
    @Published var analysisErrorMessage: String?

    private let analyzer: ChipAnalyzing
    private let sampleLoader = SampleAnalysisLoader()

    init(analyzer: ChipAnalyzing = CoreMLChipAnalyzer()) {
        self.analyzer = analyzer
    }

    func runAnalysis(
        chipConfigs: [ChipColorConfig],
        sourceLabel: String,
        previewImage: Image? = nil,
        uiImage: UIImage? = nil
    ) async {
        isAnalyzing = true
        hasProcessedImage = false
        analysisErrorMessage = nil
        captureSourceLabel = sourceLabel
        selectedPreviewImage = previewImage
        selectedPreviewImageSize = uiImage?.size
        let result = await analyzer.analyze(
            request: ChipAnalysisRequest(chipConfigs: chipConfigs, sourceLabel: sourceLabel, image: uiImage)
        )
        observations = result.observations
        draftDetections = result.detections
        captureSourceLabel = result.sourceLabel
        if result.observations.isEmpty {
            analysisErrorMessage = "沒有偵測到籌碼，請換一張角度更俯視、光線更穩定的照片。"
        }
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
            selectedPreviewImageSize = nil
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
