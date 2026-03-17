import CoreGraphics
import Foundation
import UIKit
import Vision

/// Real image-based chip analyzer using Apple's Vision framework.
///
/// Pipeline:
///   1. VNDetectContoursRequest  — find edge contours in the image
///   2. Circularity filter       — keep only roughly circular contours
///   3. Ring-region sampling     — sample pixels at 45–75 % of the radius in pixel space
///   4. Lab color matching       — compare ring mean color against each ChipColorConfig
///   5. Aggregation              — count visible tops per color, estimate stack / loose counts
struct VisionChipAnalyzer: ChipAnalyzing {

    // MARK: - Public

    func analyze(request: ChipAnalysisRequest) async -> ChipAnalysisResult {
        guard let uiImage = request.image, let cgImage = uiImage.cgImage else {
            return emptyResult(request: request)
        }

        let imgW = CGFloat(cgImage.width)
        let imgH = CGFloat(cgImage.height)

        // Step 1 + 2: detect and filter circular contours
        let candidates = detectCircles(in: cgImage, imageWidth: imgW, imageHeight: imgH)
        guard !candidates.isEmpty else {
            return emptyResult(request: request)
        }

        // Step 3: read raw pixel bytes once for fast ring sampling
        let pixelBuffer = PixelBuffer(cgImage: cgImage)

        // Precompute reference Lab for each chip config
        let configLabs = request.chipConfigs.map { hexToLab($0.colorHex) }

        // Step 4: match each candidate to the nearest chip config
        var observations: [ChipTopObservation] = []
        for candidate in candidates {
            let ringLab = pixelBuffer.sampleRingLab(
                cx: candidate.cx, cy: candidate.cy, radius: candidate.radius,
                innerRatio: 0.45, outerRatio: 0.75
            )

            var bestIdx = 0
            var bestDist = Float.greatestFiniteMagnitude
            for (idx, refLab) in configLabs.enumerated() {
                let d = deltaE(ringLab, refLab)
                if d < bestDist { bestDist = d; bestIdx = idx }
            }

            let config = request.chipConfigs[bestIdx]
            // Confidence falls off linearly: distance 0 → 1.0, distance 50+ → 0.0
            let confidence = max(0.0, Double(1.0 - bestDist / 50.0))

            let chipCandidate = ChipTopCandidate(
                center: CGPoint(x: candidate.cx, y: candidate.cy),
                normalizedCenter: CGPoint(x: candidate.cx / imgW, y: candidate.cy / imgH),
                radius: candidate.radius,
                normalizedRadius: candidate.radius / imgW,
                confidence: confidence
            )
            observations.append(ChipTopObservation(
                candidate: chipCandidate,
                predictedChipColorID: config.id,
                predictedColorName: config.name,
                confidence: confidence
            ))
        }

        // Step 5: aggregate into per-color stack estimates
        let detections = aggregateDetections(observations: observations, chipConfigs: request.chipConfigs)

        return ChipAnalysisResult(
            sourceLabel: request.sourceLabel,
            observations: observations,
            detections: detections
        )
    }

    // MARK: - Circle Detection

    private struct CircleCandidate {
        let cx: CGFloat
        let cy: CGFloat
        let radius: CGFloat
        let confidence: Double   // used for NMS sorting
    }

    private func detectCircles(
        in cgImage: CGImage,
        imageWidth: CGFloat,
        imageHeight: CGFloat
    ) -> [CircleCandidate] {

        let request = VNDetectContoursRequest()
        // Boost contrast so chip edges are more visible against the felt
        request.contrastAdjustment = 2.5
        // Most chips are on a darker background (felt / table surface)
        request.detectsDarkOnLight = false

        let handler = VNImageRequestHandler(cgImage: cgImage, options: [:])
        try? handler.perform([request])

        guard let observation = request.results?.first as? VNContoursObservation else {
            return []
        }

        var raw: [CircleCandidate] = []
        for i in 0 ..< observation.contourCount {
            guard let contour = try? observation.contour(at: i) else { continue }

            let bb = contour.normalizedPath.boundingBox
            // Convert from Vision's bottom-left normalized coords to pixel space
            let pxW = bb.width  * imageWidth
            let pxH = bb.height * imageHeight

            // Minimum chip diameter: ~30 px; skip tiny noise and full-frame boxes
            guard pxW > 30, pxH > 30,
                  pxW < imageWidth * 0.9, pxH < imageHeight * 0.9 else { continue }

            // Aspect-ratio filter: chips are circles (0.75 … 1.33)
            let aspectRatio = pxW / pxH
            guard aspectRatio >= 0.75, aspectRatio <= 1.33 else { continue }

            // Require a minimum number of path points (smooth closed curve)
            guard contour.pointCount >= 12 else { continue }

            // Vision uses bottom-left origin; flip Y for UIKit/Core Graphics
            let cx = bb.midX * imageWidth
            let cy = (1.0 - bb.midY) * imageHeight
            let radius = min(pxW, pxH) / 2.0

            // Use aspect-ratio closeness-to-1 as a proxy confidence
            let confidence = 1.0 - Double(abs(aspectRatio - 1.0))

            raw.append(CircleCandidate(cx: cx, cy: cy, radius: radius, confidence: confidence))
        }

        return nonMaxSuppress(raw, overlapThreshold: 0.4)
    }

    /// Remove overlapping circle candidates, keeping the most confident one.
    private func nonMaxSuppress(
        _ candidates: [CircleCandidate],
        overlapThreshold: Double
    ) -> [CircleCandidate] {
        let sorted = candidates.sorted { $0.confidence > $1.confidence }
        var kept: [CircleCandidate] = []
        for c in sorted {
            let dominated = kept.contains { existing in
                let dist = hypot(c.cx - existing.cx, c.cy - existing.cy)
                return Double(dist) < Double(c.radius + existing.radius) * overlapThreshold
            }
            if !dominated { kept.append(c) }
        }
        return kept
    }

    // MARK: - Aggregation

    private func aggregateDetections(
        observations: [ChipTopObservation],
        chipConfigs: [ChipColorConfig]
    ) -> [StackDetectionResult] {
        let byColor = Dictionary(grouping: observations, by: \.predictedChipColorID)

        return chipConfigs.map { config in
            let group = byColor[config.id] ?? []
            let visibleCount = group.count
            let stackCount = visibleCount > 0 ? max(1, visibleCount / max(1, config.stackSize)) : 0
            let looseCount  = visibleCount % max(1, config.stackSize)
            let avgConf = group.isEmpty ? 0.0
                : group.map(\.confidence).reduce(0, +) / Double(group.count)
            return StackDetectionResult(
                chipColorID: config.id,
                stackCount: stackCount,
                looseCount: looseCount,
                confidence: avgConf
            )
        }
    }

    // MARK: - Helpers

    private func emptyResult(request: ChipAnalysisRequest) -> ChipAnalysisResult {
        ChipAnalysisResult(
            sourceLabel: request.sourceLabel,
            observations: [],
            detections: request.chipConfigs.map {
                StackDetectionResult(chipColorID: $0.id, stackCount: 0, looseCount: 0, confidence: 0)
            }
        )
    }
}

// MARK: - PixelBuffer (ring-region sampler)

/// Wraps a CGImage's raw RGBA bytes for fast per-pixel access.
private struct PixelBuffer {
    let data: [UInt8]
    let width: Int
    let height: Int

    init(cgImage: CGImage) {
        width  = cgImage.width
        height = cgImage.height
        var buf = [UInt8](repeating: 0, count: width * height * 4)
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        if let ctx = CGContext(
            data: &buf,
            width: width, height: height,
            bitsPerComponent: 8,
            bytesPerRow: width * 4,
            space: colorSpace,
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        ) {
            ctx.draw(cgImage, in: CGRect(x: 0, y: 0, width: width, height: height))
        }
        data = buf
    }

    /// Return mean Lab color for the ring [innerRatio*r, outerRatio*r] around (cx, cy).
    func sampleRingLab(
        cx: CGFloat, cy: CGFloat, radius: CGFloat,
        innerRatio: CGFloat, outerRatio: CGFloat
    ) -> [Float] {
        let innerR = innerRatio * radius
        let outerR = outerRatio * radius

        let x0 = max(0, Int(cx - outerR))
        let x1 = min(width  - 1, Int(cx + outerR))
        let y0 = max(0, Int(cy - outerR))
        let y1 = min(height - 1, Int(cy + outerR))

        guard x0 < x1, y0 < y1 else { return [0, 0, 0] }

        var sumR: Double = 0, sumG: Double = 0, sumB: Double = 0
        var count = 0

        for py in y0 ... y1 {
            for px in x0 ... x1 {
                let dist = hypot(CGFloat(px) - cx, CGFloat(py) - cy)
                guard dist >= innerR, dist <= outerR else { continue }
                let off = (py * width + px) * 4
                guard off + 2 < data.count else { continue }
                sumR += Double(data[off])
                sumG += Double(data[off + 1])
                sumB += Double(data[off + 2])
                count += 1
            }
        }

        guard count > 0 else { return [0, 0, 0] }

        let r = Float(sumR / Double(count)) / 255.0
        let g = Float(sumG / Double(count)) / 255.0
        let b = Float(sumB / Double(count)) / 255.0
        return rgbToLab(r: r, g: g, b: b)
    }
}

// MARK: - Color conversion (sRGB → CIE L*a*b*)

private func hexToLab(_ hex: String) -> [Float] {
    let cleaned = hex.trimmingCharacters(in: CharacterSet(charactersIn: "#"))
    guard cleaned.count >= 6,
          let rv = UInt8(cleaned.prefix(2), radix: 16),
          let gv = UInt8(cleaned.dropFirst(2).prefix(2), radix: 16),
          let bv = UInt8(cleaned.dropFirst(4).prefix(2), radix: 16)
    else { return [0, 0, 0] }
    return rgbToLab(r: Float(rv) / 255.0, g: Float(gv) / 255.0, b: Float(bv) / 255.0)
}

private func rgbToLab(r: Float, g: Float, b: Float) -> [Float] {
    // sRGB → linear
    func lin(_ c: Float) -> Float {
        c <= 0.04045 ? c / 12.92 : pow((c + 0.055) / 1.055, 2.4)
    }
    let lr = lin(r), lg = lin(g), lb = lin(b)

    // Linear RGB → XYZ (D65)
    let x = lr * 0.4124564 + lg * 0.3575761 + lb * 0.1804375
    let y = lr * 0.2126729 + lg * 0.7151522 + lb * 0.0721750
    let z = lr * 0.0193339 + lg * 0.1191920 + lb * 0.9503041

    // XYZ → L*a*b* (D65 reference white)
    func f(_ t: Float) -> Float {
        let d: Float = 6.0 / 29.0
        return t > d * d * d ? cbrt(t) : t / (3.0 * d * d) + 4.0 / 29.0
    }
    let fx = f(x / 0.95047)
    let fy = f(y / 1.00000)
    let fz = f(z / 1.08883)

    return [116.0 * fy - 16.0, 500.0 * (fx - fy), 200.0 * (fy - fz)]
}

private func deltaE(_ a: [Float], _ b: [Float]) -> Float {
    guard a.count == 3, b.count == 3 else { return .greatestFiniteMagnitude }
    let dL = a[0] - b[0], da = a[1] - b[1], db = a[2] - b[2]
    return sqrt(dL * dL + da * da + db * db)
}
