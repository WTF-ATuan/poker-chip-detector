import AVFoundation
import Foundation
import UIKit

final class RealtimeCameraAnalyzer: NSObject, AVCaptureVideoDataOutputSampleBufferDelegate {
    let session = AVCaptureSession()

    private let analyzer: ChipAnalyzing
    private var chipConfigsProvider: (() -> [ChipColorConfig])?
    private var resultHandler: ((ChipAnalysisResult, UIImage) -> Void)?
    private let queue = DispatchQueue(label: "realtime.chip.analysis.queue")
    private let ciContext = CIContext()

    private var lastInferenceTime: CFTimeInterval = 0
    private var isAnalyzingFrame = false
    var minInferenceInterval: CFTimeInterval = 0.2 // 5 FPS

    init(analyzer: ChipAnalyzing) {
        self.analyzer = analyzer
        super.init()
    }

    func configureSession() -> Bool {
        session.beginConfiguration()
        session.sessionPreset = .high
        defer { session.commitConfiguration() }

        guard let camera = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back) else {
            return false
        }
        guard let input = try? AVCaptureDeviceInput(device: camera) else { return false }
        guard session.canAddInput(input) else { return false }
        session.addInput(input)

        let output = AVCaptureVideoDataOutput()
        output.alwaysDiscardsLateVideoFrames = true
        output.videoSettings = [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA]
        output.setSampleBufferDelegate(self, queue: queue)
        guard session.canAddOutput(output) else { return false }
        session.addOutput(output)

        if let connection = output.connection(with: .video), connection.isVideoOrientationSupported {
            connection.videoOrientation = .portrait
        }
        return true
    }

    func start(
        chipConfigs: @escaping () -> [ChipColorConfig],
        onResult: @escaping (ChipAnalysisResult, UIImage) -> Void
    ) {
        chipConfigsProvider = chipConfigs
        resultHandler = onResult
        if !session.isRunning {
            session.startRunning()
        }
    }

    func stop() {
        if session.isRunning {
            session.stopRunning()
        }
    }

    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        let now = CACurrentMediaTime()
        guard now - lastInferenceTime >= minInferenceInterval else { return }
        guard !isAnalyzingFrame else { return }
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        guard let chipConfigsProvider, let resultHandler else { return }

        isAnalyzingFrame = true
        lastInferenceTime = now

        let image = convertToUIImage(pixelBuffer: pixelBuffer)
        Task {
            let result = await analyzer.analyze(
                request: ChipAnalysisRequest(
                    chipConfigs: chipConfigsProvider(),
                    sourceLabel: "Realtime camera",
                    image: image
                )
            )
            DispatchQueue.main.async {
                resultHandler(result, image)
            }
            self.isAnalyzingFrame = false
        }
    }

    private func convertToUIImage(pixelBuffer: CVPixelBuffer) -> UIImage {
        let ciImage = CIImage(cvPixelBuffer: pixelBuffer)
        guard let cgImage = ciContext.createCGImage(ciImage, from: ciImage.extent) else {
            return UIImage()
        }
        return UIImage(cgImage: cgImage)
    }
}
