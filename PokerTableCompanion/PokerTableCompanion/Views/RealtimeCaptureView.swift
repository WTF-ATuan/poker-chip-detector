import AVFoundation
import SwiftUI

struct RealtimeCaptureView: View {
    @EnvironmentObject private var appState: AppState
    @StateObject private var viewModel = RealtimeCaptureViewModel()

    var body: some View {
        VStack(spacing: 16) {
            ZStack {
                CameraPreviewView(session: viewModel.session)
                    .overlay(Color.black.opacity(0.08))

                GeometryReader { proxy in
                    ForEach(viewModel.observations) { observation in
                        Circle()
                            .stroke(color(for: observation), lineWidth: 2)
                            .frame(
                                width: max(20, observation.candidate.normalizedRadius * proxy.size.width * 2),
                                height: max(20, observation.candidate.normalizedRadius * proxy.size.width * 2)
                            )
                            .position(
                                x: observation.candidate.normalizedCenter.x * proxy.size.width,
                                y: observation.candidate.normalizedCenter.y * proxy.size.height
                            )
                    }
                }
            }
            .clipShape(RoundedRectangle(cornerRadius: 18, style: .continuous))
            .frame(height: 320)

            Text(viewModel.statusText)
                .font(.footnote)
                .foregroundStyle(.secondary)

            if let msg = viewModel.permissionMessage {
                Text(msg)
                    .font(.footnote)
                    .foregroundStyle(.red)
            }

            HStack(spacing: 12) {
                Button("Start Realtime") {
                    viewModel.start { appState.chipConfigs }
                }
                .buttonStyle(.borderedProminent)

                Button("Stop") {
                    viewModel.stop()
                }
                .buttonStyle(.bordered)
            }

            Button("Apply Current Estimate") {
                appState.applyDetections(viewModel.detections)
            }
            .buttonStyle(.borderedProminent)
            .disabled(viewModel.detections.isEmpty)

            Spacer()
        }
        .padding(20)
        .navigationTitle("Realtime Capture")
        .navigationBarTitleDisplayMode(.inline)
        .onDisappear { viewModel.stop() }
    }

    private func color(for observation: ChipTopObservation) -> Color {
        if let config = appState.chipConfigs.first(where: { $0.id == observation.predictedChipColorID }) {
            return config.swiftUIColor
        }
        return .gray
    }
}

private struct CameraPreviewView: UIViewRepresentable {
    let session: AVCaptureSession

    func makeUIView(context: Context) -> PreviewUIView {
        let view = PreviewUIView()
        view.previewLayer.videoGravity = .resizeAspectFill
        view.previewLayer.session = session
        return view
    }

    func updateUIView(_ uiView: PreviewUIView, context: Context) {
        uiView.previewLayer.session = session
    }
}

private final class PreviewUIView: UIView {
    override class var layerClass: AnyClass { AVCaptureVideoPreviewLayer.self }
    var previewLayer: AVCaptureVideoPreviewLayer { layer as! AVCaptureVideoPreviewLayer }
}
