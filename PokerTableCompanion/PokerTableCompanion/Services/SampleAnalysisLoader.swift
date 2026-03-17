import CoreGraphics
import Foundation
import SwiftUI
import UIKit

struct SampleAnalysisLoader {
    enum LoaderError: Error {
        case missingImage
        case missingJSON
    }

    func loadSampleCapture(chipConfigs: [ChipColorConfig]) throws -> (image: Image, result: ChipAnalysisResult) {
        guard let imageURL = Bundle.main.url(forResource: "sample_capture", withExtension: "jpg") else {
            throw LoaderError.missingImage
        }
        guard let jsonURL = Bundle.main.url(forResource: "sample_capture_analysis", withExtension: "json") else {
            throw LoaderError.missingJSON
        }

        let image = Image(uiImage: UIImage(contentsOfFile: imageURL.path) ?? UIImage())
        let data = try Data(contentsOf: jsonURL)
        let decoded = try JSONDecoder().decode(PipelineSampleAnalysis.self, from: data)
        let result = decoded.toChipAnalysisResult(chipConfigs: chipConfigs)
        return (image, result)
    }
}

private struct PipelineSampleAnalysis: Decodable {
    struct Observation: Decodable {
        struct NormalizedCenter: Decodable {
            let x: CGFloat
            let y: CGFloat
        }

        let normalized_center: NormalizedCenter
        let normalized_radius: CGFloat
        let candidate_confidence: Double
        let predicted_color: String
        let color_confidence: Double
    }

    let observations: [Observation]

    func toChipAnalysisResult(chipConfigs: [ChipColorConfig]) -> ChipAnalysisResult {
        let observations = observations.map { entry -> ChipTopObservation in
            let matchedConfig = chipConfigs.first {
                $0.name.caseInsensitiveCompare(entry.predicted_color) == .orderedSame
            } ?? chipConfigs.first ?? ChipColorConfig(name: entry.predicted_color.capitalized, denomination: 100, colorHex: "#B6B6B6")

            let candidate = ChipTopCandidate(
                center: .zero,
                normalizedCenter: CGPoint(x: entry.normalized_center.x, y: entry.normalized_center.y),
                radius: 0,
                normalizedRadius: entry.normalized_radius,
                confidence: entry.candidate_confidence
            )
            return ChipTopObservation(
                candidate: candidate,
                predictedChipColorID: matchedConfig.id,
                predictedColorName: matchedConfig.name,
                confidence: entry.color_confidence
            )
        }

        let grouped = Dictionary(grouping: observations, by: \.predictedChipColorID)
        let detections = grouped.compactMap { colorID, entries -> StackDetectionResult? in
            guard let config = chipConfigs.first(where: { $0.id == colorID }) else { return nil }
            return StackDetectionResult(
                chipColorID: config.id,
                stackCount: entries.count,
                looseCount: 0,
                confidence: entries.map(\.confidence).reduce(0, +) / Double(entries.count)
            )
        }
        .sorted { lhs, rhs in lhs.confidence > rhs.confidence }

        return ChipAnalysisResult(
            sourceLabel: "Loaded calibrated sample",
            observations: observations,
            detections: detections
        )
    }
}
