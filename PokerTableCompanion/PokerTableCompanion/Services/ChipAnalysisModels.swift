import CoreGraphics
import Foundation
import UIKit

struct ChipAnalysisRequest {
    var chipConfigs: [ChipColorConfig]
    var sourceLabel: String
    var image: UIImage?
}

struct ChipTopCandidate: Identifiable, Hashable {
    let id: UUID
    var center: CGPoint
    var normalizedCenter: CGPoint
    var radius: CGFloat
    var normalizedRadius: CGFloat
    var confidence: Double

    init(
        id: UUID = UUID(),
        center: CGPoint,
        normalizedCenter: CGPoint,
        radius: CGFloat,
        normalizedRadius: CGFloat,
        confidence: Double
    ) {
        self.id = id
        self.center = center
        self.normalizedCenter = normalizedCenter
        self.radius = radius
        self.normalizedRadius = normalizedRadius
        self.confidence = confidence
    }
}

struct ChipTopObservation: Identifiable, Hashable {
    let id: UUID
    var candidate: ChipTopCandidate
    var predictedChipColorID: UUID
    var predictedColorName: String
    var visibleTopCountContribution: Int
    var confidence: Double
    var rawModelClassLabel: String?
    var rawModelClassConfidence: Double?

    init(
        id: UUID = UUID(),
        candidate: ChipTopCandidate,
        predictedChipColorID: UUID,
        predictedColorName: String,
        visibleTopCountContribution: Int = 1,
        confidence: Double,
        rawModelClassLabel: String? = nil,
        rawModelClassConfidence: Double? = nil
    ) {
        self.id = id
        self.candidate = candidate
        self.predictedChipColorID = predictedChipColorID
        self.predictedColorName = predictedColorName
        self.visibleTopCountContribution = visibleTopCountContribution
        self.confidence = confidence
        self.rawModelClassLabel = rawModelClassLabel
        self.rawModelClassConfidence = rawModelClassConfidence
    }
}

struct ChipAnalysisResult {
    var sourceLabel: String
    var observations: [ChipTopObservation]
    var detections: [StackDetectionResult]
    var debugStats: AnalysisDebugStats = .empty
}

struct AnalysisDebugStats {
    var decodedCount: Int
    var afterNMSCount: Int
    var afterDedupCount: Int

    static let empty = AnalysisDebugStats(
        decodedCount: 0,
        afterNMSCount: 0,
        afterDedupCount: 0
    )
}
