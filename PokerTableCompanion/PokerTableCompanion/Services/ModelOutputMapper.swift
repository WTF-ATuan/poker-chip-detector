import CoreGraphics
import Foundation

struct DecodedDetection {
    var normalizedRect: CGRect
    var classIndex: Int
    var classLabel: String
    var confidence: Double
}

enum ModelOutputMapper {
    // Matches color-first training dataset.yaml class order.
    static let classLabels: [String] = ["red", "pink", "green", "black"]

    static func decodeYOLOOutput(
        _ values: UnsafeBufferPointer<Float>,
        featureChannels: Int,
        anchors: Int,
        confidenceThreshold: Float
    ) -> [DecodedDetection] {
        guard featureChannels >= 5 else { return [] }

        let classCount = featureChannels - 4
        var detections: [DecodedDetection] = []
        detections.reserveCapacity(anchors / 4)

        for anchor in 0 ..< anchors {
            let x = values[(0 * anchors) + anchor]
            let y = values[(1 * anchors) + anchor]
            let w = values[(2 * anchors) + anchor]
            let h = values[(3 * anchors) + anchor]

            var bestClass = 0
            var bestScore: Float = 0
            for c in 0 ..< classCount {
                // CoreML-exported YOLO heads can output logits; convert to probability.
                let raw = values[((4 + c) * anchors) + anchor]
                let score = sigmoid(raw)
                if score > bestScore {
                    bestScore = score
                    bestClass = c
                }
            }

            guard bestScore >= confidenceThreshold else { continue }
            guard w > 0, h > 0 else { continue }

            let x1 = max(0, x - w * 0.5)
            let y1 = max(0, y - h * 0.5)
            let x2 = min(640, x + w * 0.5)
            let y2 = min(640, y + h * 0.5)
            let rect = CGRect(
                x: CGFloat(x1 / 640.0),
                y: CGFloat(y1 / 640.0),
                width: CGFloat((x2 - x1) / 640.0),
                height: CGFloat((y2 - y1) / 640.0)
            )
            // Filter obvious noise boxes (too tiny / too huge).
            guard rect.width > 0.045, rect.height > 0.045 else { continue }
            guard rect.width < 0.55, rect.height < 0.55 else { continue }

            detections.append(
                DecodedDetection(
                    normalizedRect: rect,
                    classIndex: bestClass,
                    classLabel: classLabel(for: bestClass),
                    confidence: Double(bestScore)
                )
            )
        }
        return detections
    }

    static func nonMaxSuppression(_ detections: [DecodedDetection], iouThreshold: CGFloat) -> [DecodedDetection] {
        let sorted = detections.sorted { $0.confidence > $1.confidence }
        var kept: [DecodedDetection] = []
        kept.reserveCapacity(sorted.count)

        for candidate in sorted {
            let overlaps = kept.contains { existing in
                iou(existing.normalizedRect, candidate.normalizedRect) >= iouThreshold
            }
            if !overlaps {
                kept.append(candidate)
            }
        }
        return kept
    }

    static func deduplicateByCenterAndRadius(_ detections: [DecodedDetection]) -> [DecodedDetection] {
        let sorted = detections.sorted { $0.confidence > $1.confidence }
        var kept: [DecodedDetection] = []
        kept.reserveCapacity(sorted.count)

        for candidate in sorted {
            let candidateCenter = CGPoint(
                x: candidate.normalizedRect.midX,
                y: candidate.normalizedRect.midY
            )
            let candidateRadius = max(
                candidate.normalizedRect.width,
                candidate.normalizedRect.height
            ) * 0.5

            let isDuplicate = kept.contains { existing in
                let existingCenter = CGPoint(
                    x: existing.normalizedRect.midX,
                    y: existing.normalizedRect.midY
                )
                let existingRadius = max(
                    existing.normalizedRect.width,
                    existing.normalizedRect.height
                ) * 0.5

                let dx = existingCenter.x - candidateCenter.x
                let dy = existingCenter.y - candidateCenter.y
                let centerDistance = sqrt((dx * dx) + (dy * dy))

                let bigger = max(existingRadius, candidateRadius)
                let smaller = max(0.0001, min(existingRadius, candidateRadius))
                let radiusRatio = bigger / smaller

                // Treat as duplicate if centers are very close and sizes are similar.
                return centerDistance < (0.60 * bigger) && radiusRatio < 1.35
            }

            if !isDuplicate {
                kept.append(candidate)
            }
        }

        return kept
    }

    static func mapDetectionToChipConfig(
        _ detection: DecodedDetection,
        chipConfigs: [ChipColorConfig]
    ) -> ChipColorConfig? {
        if chipConfigs.isEmpty { return nil }

        // 1) Exact color-name match (current color-first model path)
        if let byName = chipConfigs.first(where: { $0.name.caseInsensitiveCompare(detection.classLabel) == .orderedSame }) {
            return byName
        }

        // 2) Tolerant contains-match for config names (e.g. "Red 1000")
        if let byContains = chipConfigs.first(where: {
            $0.name.lowercased().contains(detection.classLabel.lowercased())
        }) {
            return byContains
        }

        // 3) Legacy numeric class label -> closest denomination (kept for compatibility)
        if let denom = Int(detection.classLabel) {
            return chipConfigs.min(by: { abs($0.denomination - denom) < abs($1.denomination - denom) })
        }

        // 4) Safe fallback
        return chipConfigs.first
    }

    static func classLabel(for classIndex: Int) -> String {
        guard classIndex >= 0, classIndex < classLabels.count else { return "unknown" }
        return classLabels[classIndex]
    }

    private static func iou(_ a: CGRect, _ b: CGRect) -> CGFloat {
        let intersection = a.intersection(b)
        if intersection.isNull { return 0 }
        let interArea = intersection.width * intersection.height
        let unionArea = (a.width * a.height) + (b.width * b.height) - interArea
        guard unionArea > 0 else { return 0 }
        return interArea / unionArea
    }

    private static func sigmoid(_ x: Float) -> Float {
        1.0 / (1.0 + exp(-x))
    }
}
