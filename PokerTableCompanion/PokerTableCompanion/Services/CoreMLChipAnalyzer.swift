import CoreGraphics
import CoreML
import Foundation
import UIKit
import Vision

struct CoreMLChipAnalyzer: ChipAnalyzing {
    private let confidenceThreshold: Float = 0.45
    private let nmsIoUThreshold: CGFloat = 0.45
    private let maxDetections = 80
    private let legacyAnalyzer = VisionChipAnalyzer()

    func analyze(request: ChipAnalysisRequest) async -> ChipAnalysisResult {
        guard let uiImage = request.image, let cgImage = uiImage.cgImage else {
            return emptyResult(request: request, stage: "no-image")
        }
        guard let vnModel = Self.loadVisionModel() else {
            return emptyResult(request: request, stage: "model-missing")
        }

        let handler = VNImageRequestHandler(cgImage: cgImage, options: [:])
        let req = VNCoreMLRequest(model: vnModel)
        req.imageCropAndScaleOption = .scaleFill

        do {
            try handler.perform([req])
            guard let observation = req.results?.first as? VNCoreMLFeatureValueObservation,
                  let multiArray = observation.featureValue.multiArrayValue else {
                return emptyResult(request: request, stage: "coreml-output-missing")
            }

            let imgW = CGFloat(cgImage.width)
            let imgH = CGFloat(cgImage.height)
            let pixels = rgbaPixels(from: cgImage)

            let decoded = decode(multiArray: multiArray)
            let nmsFiltered = Array(
                ModelOutputMapper.nonMaxSuppression(decoded, iouThreshold: nmsIoUThreshold)
                    .prefix(maxDetections)
            )
            let filtered = ModelOutputMapper.deduplicateByCenterAndRadius(nmsFiltered)
            let debugStats = AnalysisDebugStats(
                decodedCount: decoded.count,
                afterNMSCount: nmsFiltered.count,
                afterDedupCount: filtered.count
            )

            let observations: [ChipTopObservation] = filtered.compactMap { (det: DecodedDetection) -> ChipTopObservation? in
                // Ignore common table-pattern false positives in top area.
                if det.normalizedRect.midY < 0.33 {
                    return nil
                }

                let regionColor = sampleRegionColor(
                    det: det,
                    pixels: pixels,
                    imageWidth: Int(imgW),
                    imageHeight: Int(imgH)
                )
                let colorName = classifyChipColor(regionColor)

                let config = mapColorNameToConfig(colorName, configs: request.chipConfigs)
                    ?? ModelOutputMapper.mapDetectionToChipConfig(det, chipConfigs: request.chipConfigs)
                guard let config else {
                    return nil
                }

                let cxNorm = det.normalizedRect.midX
                let cyNorm = det.normalizedRect.midY
                let rNorm = min(det.normalizedRect.width, det.normalizedRect.height) * 0.5

                let candidate = ChipTopCandidate(
                    center: CGPoint(x: cxNorm * imgW, y: cyNorm * imgH),
                    normalizedCenter: CGPoint(x: cxNorm, y: cyNorm),
                    radius: rNorm * imgW,
                    normalizedRadius: rNorm,
                    confidence: det.confidence
                )
                return ChipTopObservation(
                    candidate: candidate,
                    predictedChipColorID: config.id,
                    predictedColorName: colorName ?? config.name,
                    confidence: det.confidence,
                    rawModelClassLabel: det.classLabel,
                    rawModelClassConfidence: det.confidence
                )
            }

            if observations.isEmpty {
                // Fallback 1: previous Vision-based analyzer.
                let legacy = await legacyAnalyzer.analyze(request: request)
                if !legacy.observations.isEmpty {
                    var tagged = legacy
                    tagged.sourceLabel = request.sourceLabel + " (vision-fallback)"
                    tagged.debugStats = debugStats
                    return tagged
                }

                // Fallback 2: color-only estimation so UI still shows color tags.
                let colorOnly = colorOnlyFallback(request: request, cgImage: cgImage)
                if !colorOnly.observations.isEmpty {
                    var tagged = colorOnly
                    tagged.debugStats = debugStats
                    return tagged
                }
                var tagged = colorOnly
                tagged.debugStats = debugStats
                return tagged
            }

            let detections = aggregate(observations: observations, chipConfigs: request.chipConfigs)
            return ChipAnalysisResult(
                sourceLabel: request.sourceLabel + " (coreml)",
                observations: observations,
                detections: detections,
                debugStats: debugStats
            )
        } catch {
            // If CoreML request throws, still try color-only fallback before declaring failure.
            let colorOnly = colorOnlyFallback(request: request, cgImage: cgImage)
            if !colorOnly.observations.isEmpty {
                return colorOnly
            }
            return emptyResult(request: request, stage: "coreml-exception")
        }
    }

    private func decode(multiArray: MLMultiArray) -> [DecodedDetection] {
        let shape = multiArray.shape.map { Int(truncating: $0) }
        guard shape.count == 3 else { return [] }

        let channels = shape[1]
        let anchors = shape[2]
        let ptr = multiArray.dataPointer.bindMemory(to: Float.self, capacity: channels * anchors)
        let buffer = UnsafeBufferPointer(start: ptr, count: channels * anchors)

        return ModelOutputMapper.decodeYOLOOutput(
            buffer,
            featureChannels: channels,
            anchors: anchors,
            confidenceThreshold: confidenceThreshold
        )
    }

    private func aggregate(observations: [ChipTopObservation], chipConfigs: [ChipColorConfig]) -> [StackDetectionResult] {
        let byColor = Dictionary(grouping: observations, by: \.predictedChipColorID)
        return chipConfigs.map { cfg in
            let group = byColor[cfg.id] ?? []
            let visible = group.count
            let stacks = visible > 0 ? visible / max(1, cfg.stackSize) : 0
            let loose = visible % max(1, cfg.stackSize)
            let confidence = group.isEmpty ? 0.0 : group.map(\.confidence).reduce(0, +) / Double(group.count)
            return StackDetectionResult(
                chipColorID: cfg.id,
                stackCount: stacks,
                looseCount: loose,
                confidence: confidence
            )
        }
    }

    private func emptyResult(request: ChipAnalysisRequest, stage: String) -> ChipAnalysisResult {
        ChipAnalysisResult(
            sourceLabel: request.sourceLabel + " (\(stage))",
            observations: [],
            detections: request.chipConfigs.map {
                StackDetectionResult(chipColorID: $0.id, stackCount: 0, looseCount: 0, confidence: 0)
            }
        )
    }

    private static func loadVisionModel() -> VNCoreMLModel? {
        do {
            // In app bundle, Xcode typically compiles CoreML assets into .mlmodelc.
            if let compiledURL = Bundle.main.url(forResource: "best", withExtension: "mlmodelc") {
                let mlModel = try MLModel(contentsOf: compiledURL)
                return try VNCoreMLModel(for: mlModel)
            }

            // Dev fallback: load .mlpackage and compile at runtime.
            if let packageURL = Bundle.main.url(forResource: "best", withExtension: "mlpackage") {
                let compiled = try MLModel.compileModel(at: packageURL)
                let mlModel = try MLModel(contentsOf: compiled)
                return try VNCoreMLModel(for: mlModel)
            }
            return nil
        } catch {
            return nil
        }
    }

    private func colorOnlyFallback(request: ChipAnalysisRequest, cgImage: CGImage) -> ChipAnalysisResult {
        guard let pixelData = rgbaPixels(from: cgImage) else {
            return emptyResult(request: request, stage: "color-fallback-no-pixels")
        }
        let width = cgImage.width
        let height = cgImage.height
        guard width > 0, height > 0 else {
            return emptyResult(request: request, stage: "color-fallback-size-invalid")
        }

        var redCount = 0
        var pinkCount = 0
        var greenCount = 0
        var sampleCount = 0

        // Focus on lower 2/3 of the image where chips usually appear.
        let yStart = height / 3
        let step = max(2, min(width, height) / 300)
        for y in stride(from: yStart, to: height, by: step) {
            for x in stride(from: 0, to: width, by: step) {
                let idx = (y * width + x) * 4
                guard idx + 2 < pixelData.count else { continue }
                let r = Double(pixelData[idx])
                let g = Double(pixelData[idx + 1])
                let b = Double(pixelData[idx + 2])
                sampleCount += 1

            let maxV = max(r, max(g, b))
            let minV = min(r, min(g, b))
            let saturation = maxV > 0 ? (maxV - minV) / maxV : 0
            let brightness = maxV / 255.0

            if r > g * 1.20, r > b * 1.08, r > 80 {
                // Separate stronger red/orange from lighter pink by saturation + brightness.
                if saturation < 0.42, brightness > 0.45 {
                    pinkCount += 1
                } else {
                    redCount += 1
                }
                } else if g > r * 1.15, g > b * 1.15, g > 70 {
                    greenCount += 1
                }
            }
        }

        guard sampleCount > 0 else {
            return emptyResult(request: request, stage: "color-fallback-no-samples")
        }

        let observations = buildColorOnlyObservations(
            request: request,
            counts: [
                ("red", redCount),
                ("pink", pinkCount),
                ("green", greenCount),
            ],
            sampleCount: sampleCount
        )
        let detections = aggregate(observations: observations, chipConfigs: request.chipConfigs)
        return ChipAnalysisResult(
            sourceLabel: request.sourceLabel + " (color fallback)",
            observations: observations,
            detections: detections
        )
    }

    private func buildColorOnlyObservations(
        request: ChipAnalysisRequest,
        counts: [(String, Int)],
        sampleCount: Int
    ) -> [ChipTopObservation] {
        var observations: [ChipTopObservation] = []
        let colorSlots: [CGPoint] = [
            CGPoint(x: 0.30, y: 0.70),
            CGPoint(x: 0.50, y: 0.70),
            CGPoint(x: 0.70, y: 0.70),
        ]

        var slot = 0
        for (name, count) in counts {
            let ratio = Double(count) / Double(sampleCount)
            guard ratio > 0.004 else { continue } // keep loose to avoid "all fail" experience

            guard let config = mapColorNameToConfig(name, configs: request.chipConfigs) else { continue }
            let center = colorSlots[min(slot, colorSlots.count - 1)]
            slot += 1

            let candidate = ChipTopCandidate(
                center: .zero,
                normalizedCenter: center,
                radius: 0,
                normalizedRadius: 0.08,
                confidence: min(0.95, max(0.45, ratio * 6.0))
            )
            observations.append(
                ChipTopObservation(
                    candidate: candidate,
                    predictedChipColorID: config.id,
                    predictedColorName: displayColorName(name, configName: config.name),
                    confidence: candidate.confidence
                )
            )
        }
        return observations
    }

    private func mapColorNameToConfig(_ colorName: String, configs: [ChipColorConfig]) -> ChipColorConfig? {
        switch colorName {
        case "red":
            return configs.first(where: { $0.name.lowercased().contains("red") })
                ?? configs.first(where: { $0.name.lowercased().contains("orange") })
                ?? configs.first
        case "pink":
            return configs.first(where: { $0.name.lowercased().contains("pink") }) ?? configs.first
        case "green":
            return configs.first(where: { $0.name.lowercased().contains("green") }) ?? configs.first
        default:
            return configs.first
        }
    }

    private func mapColorNameToConfig(_ colorName: String?, configs: [ChipColorConfig]) -> ChipColorConfig? {
        guard let colorName else { return nil }
        return mapColorNameToConfig(colorName, configs: configs)
    }

    private func displayColorName(_ detected: String, configName: String) -> String {
        if detected == "red" { return "Red" }
        if detected == "pink" { return "Pink" }
        if detected == "green" { return "Green" }
        return configName
    }

    private func rgbaPixels(from cgImage: CGImage) -> [UInt8]? {
        let width = cgImage.width
        let height = cgImage.height
        var pixels = [UInt8](repeating: 0, count: width * height * 4)
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        guard let context = CGContext(
            data: &pixels,
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: width * 4,
            space: colorSpace,
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        ) else {
            return nil
        }
        context.draw(cgImage, in: CGRect(x: 0, y: 0, width: width, height: height))
        return pixels
    }

    private func sampleRegionColor(
        det: DecodedDetection,
        pixels: [UInt8]?,
        imageWidth: Int,
        imageHeight: Int
    ) -> (r: Double, g: Double, b: Double)? {
        guard let pixels, imageWidth > 0, imageHeight > 0 else { return nil }

        let cx = Int(det.normalizedRect.midX * CGFloat(imageWidth))
        let cy = Int(det.normalizedRect.midY * CGFloat(imageHeight))
        let radius = max(4, Int(min(det.normalizedRect.width, det.normalizedRect.height) * CGFloat(imageWidth) * 0.28))

        var sr = 0.0
        var sg = 0.0
        var sb = 0.0
        var count = 0.0

        let x0 = max(0, cx - radius)
        let y0 = max(0, cy - radius)
        let x1 = min(imageWidth - 1, cx + radius)
        let y1 = min(imageHeight - 1, cy + radius)

        for y in stride(from: y0, through: y1, by: 2) {
            for x in stride(from: x0, through: x1, by: 2) {
                let dx = x - cx
                let dy = y - cy
                let dist2 = dx * dx + dy * dy
                if dist2 > radius * radius { continue }
                // Prefer a ring region to reduce center-text and highlight pollution.
                if dist2 < Int(Double(radius * radius) * 0.20) { continue }

                let idx = (y * imageWidth + x) * 4
                if idx + 2 >= pixels.count { continue }
                sr += Double(pixels[idx])
                sg += Double(pixels[idx + 1])
                sb += Double(pixels[idx + 2])
                count += 1
            }
        }

        guard count > 0 else { return nil }
        return (sr / count, sg / count, sb / count)
    }

    private func classifyChipColor(_ rgb: (r: Double, g: Double, b: Double)?) -> String? {
        guard let rgb else { return nil }
        let r = rgb.r / 255.0
        let g = rgb.g / 255.0
        let b = rgb.b / 255.0

        let maxV = max(r, max(g, b))
        let minV = min(r, min(g, b))
        let delta = maxV - minV
        let brightness = maxV
        let saturation = maxV > 0 ? delta / maxV : 0

        var hue: Double = 0
        if delta > 0.0001 {
            if maxV == r {
                hue = ((g - b) / delta).truncatingRemainder(dividingBy: 6.0)
            } else if maxV == g {
                hue = ((b - r) / delta) + 2.0
            } else {
                hue = ((r - g) / delta) + 4.0
            }
            hue /= 6.0
            if hue < 0 { hue += 1.0 }
        }

        if brightness < 0.22 {
            return "black"
        }
        if saturation < 0.16 {
            return nil
        }

        // Red/orange chips
        if hue < 0.08 || hue > 0.95 {
            return "red"
        }
        // Pink chips (light red/magenta)
        if (hue >= 0.86 && hue <= 0.95) || (r > 0.55 && b > 0.35 && saturation < 0.55) {
            return "pink"
        }
        // Green chips
        if hue >= 0.22 && hue <= 0.45 {
            return "green"
        }
        return nil
    }
}
