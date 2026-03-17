import Foundation

struct MockChipAnalyzer: ChipAnalyzing {
    func analyze(request: ChipAnalysisRequest) async -> ChipAnalysisResult {
        let observations = request.chipConfigs.enumerated().flatMap { index, config -> [ChipTopObservation] in
            let count = max(1, 3 - index)
            return (0..<count).map { step in
                let center = CGPoint(x: 90 + (step * 56) + (index * 18), y: 120 + (index * 52))
                let normalizedCenter = CGPoint(
                    x: 0.18 + CGFloat(step) * 0.15 + CGFloat(index) * 0.04,
                    y: 0.28 + CGFloat(index) * 0.12
                )
                let radius = CGFloat(32 + max(0, 4 - index))
                let candidate = ChipTopCandidate(
                    center: center,
                    normalizedCenter: normalizedCenter,
                    radius: radius,
                    normalizedRadius: radius / 320.0,
                    confidence: max(0.55, 0.9 - Double(index) * 0.08)
                )
                return ChipTopObservation(
                    candidate: candidate,
                    predictedChipColorID: config.id,
                    predictedColorName: config.name,
                    confidence: candidate.confidence
                )
            }
        }

        let detections = request.chipConfigs.enumerated().map { index, config in
            StackDetectionResult(
                chipColorID: config.id,
                stackCount: max(0, 3 - index),
                looseCount: index == 0 ? 4 : index == 1 ? 2 : 0,
                confidence: max(0.54, 0.92 - Double(index) * 0.1)
            )
        }

        return ChipAnalysisResult(
            sourceLabel: request.sourceLabel,
            observations: observations,
            detections: detections
        )
    }
}
