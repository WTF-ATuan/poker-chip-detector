import AVFoundation
import Foundation
import SwiftUI
import UIKit

@MainActor
final class RealtimeCaptureViewModel: ObservableObject {
    @Published var observations: [ChipTopObservation] = []
    @Published var detections: [StackDetectionResult] = []
    @Published var statusText = "Camera idle"
    @Published var latestPreviewImage: Image?
    @Published var hasCameraPermission = false
    @Published var permissionMessage: String?

    let cameraAnalyzer: RealtimeCameraAnalyzer
    var session: AVCaptureSession { cameraAnalyzer.session }

    init(analyzer: ChipAnalyzing = CoreMLChipAnalyzer()) {
        self.cameraAnalyzer = RealtimeCameraAnalyzer(analyzer: analyzer)
    }

    func start(chipConfigs: @escaping () -> [ChipColorConfig]) {
        AVCaptureDevice.requestAccess(for: .video) { [weak self] granted in
            guard let self else { return }
            Task { @MainActor in
                self.hasCameraPermission = granted
                if !granted {
                    self.permissionMessage = "Camera permission is required for realtime analysis."
                    return
                }
                if self.cameraAnalyzer.configureSession() == false {
                    self.permissionMessage = "Failed to configure camera session."
                    return
                }
                self.statusText = "Realtime analysis started"
                self.cameraAnalyzer.start(chipConfigs: chipConfigs) { [weak self] result, image in
                    guard let self else { return }
                    self.observations = result.observations
                    self.detections = result.detections
                    self.latestPreviewImage = Image(uiImage: image)
                    self.statusText = "Realtime: \(result.observations.count) chips detected"
                }
            }
        }
    }

    func stop() {
        cameraAnalyzer.stop()
        statusText = "Realtime analysis stopped"
    }
}
