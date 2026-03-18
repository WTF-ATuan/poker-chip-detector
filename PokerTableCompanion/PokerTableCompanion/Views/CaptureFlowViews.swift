import PhotosUI
import SwiftUI

struct CaptureIntroView: View {
    @EnvironmentObject private var appState: AppState
    @StateObject private var viewModel = CaptureFlowViewModel()
    @State private var selectedItem: PhotosPickerItem?
    @State private var showingCamera = false

    var body: some View {
        ScrollView {
            VStack(spacing: 20) {
                instructionCard
                captureMethods
                previewCard
                if viewModel.isAnalyzing {
                    ProgressView("Analyzing image…")
                        .padding(.top, 12)
                }

                NavigationLink {
                    DetectionReviewView(viewModel: viewModel)
                } label: {
                    Text("Review estimated stacks")
                        .frame(maxWidth: .infinity)
                }
                .buttonStyle(.borderedProminent)
                .disabled(!viewModel.hasProcessedImage)
            }
            .padding(20)
        }
        .background(AppTheme.background.ignoresSafeArea())
        .navigationTitle("Capture Stack")
        .navigationBarTitleDisplayMode(.inline)
        .toolbar {
            ToolbarItem(placement: .topBarTrailing) {
                NavigationLink {
                    HandRecorderView()
                } label: {
                    Image(systemName: "square.and.pencil")
                }
            }
        }
        .sheet(isPresented: $showingCamera) {
            CameraPicker { image in
                guard let image else { return }
                Task {
                    await viewModel.runAnalysis(
                        chipConfigs: appState.chipConfigs,
                        sourceLabel: "Captured from camera",
                        previewImage: Image(uiImage: image),
                        uiImage: image
                    )
                }
            }
            .ignoresSafeArea()
        }
        .task(id: selectedItem) {
            guard let selectedItem else { return }
            let imageData = try? await selectedItem.loadTransferable(type: Data.self)
            let uiImage = imageData.flatMap(UIImage.init(data:))
            await viewModel.runAnalysis(
                chipConfigs: appState.chipConfigs,
                sourceLabel: "Imported from photo library",
                previewImage: uiImage.map(Image.init(uiImage:)),
                uiImage: uiImage
            )
        }
    }

    private var instructionCard: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("How to capture")
                .font(.title3.weight(.semibold))
            Text("目前先專注在照片辨識。請盡量俯拍、保持光線均勻，並讓籌碼完整出現在畫面中。")
                .foregroundStyle(.secondary)
        }
        .cardStyle()
    }

    private var captureMethods: some View {
        VStack(spacing: 12) {
            Button {
                showingCamera = true
            } label: {
                Label("Capture with camera", systemImage: "camera.fill")
                    .frame(maxWidth: .infinity)
            }
            .buttonStyle(.borderedProminent)

            PhotosPicker(selection: $selectedItem, matching: .images) {
                Label("Choose from library", systemImage: "photo.on.rectangle")
                    .frame(maxWidth: .infinity)
            }
            .buttonStyle(.bordered)

        }
    }

    private var previewCard: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Selected Input")
                .font(.title3.weight(.semibold))
            Group {
                if let preview = viewModel.selectedPreviewImage {
                    ChipPreviewOverlay(
                        preview: preview,
                        observations: viewModel.observations,
                        chipConfigs: appState.chipConfigs,
                        imageSize: viewModel.selectedPreviewImageSize
                    )
                    .frame(height: 240)
                } else {
                    RoundedRectangle(cornerRadius: 18, style: .continuous)
                        .fill(AppTheme.backgroundRaised)
                        .frame(height: 180)
                        .overlay(
                            Image(systemName: "camera.macro")
                                .font(.system(size: 40))
                                .foregroundStyle(.secondary)
                        )
                }
            }
            Text(viewModel.captureSourceLabel)
                .font(.footnote)
                .foregroundStyle(.secondary)

            if let analysisErrorMessage = viewModel.analysisErrorMessage {
                Text(analysisErrorMessage)
                    .font(.footnote)
                    .foregroundStyle(.red)
            }

            debugPanel

            if !viewModel.observations.isEmpty {
                ScrollView(.horizontal, showsIndicators: false) {
                    HStack(spacing: 8) {
                        ForEach(groupedObservationLabels, id: \.self) { label in
                            Text(label)
                                .font(.caption.weight(.semibold))
                                .padding(.horizontal, 10)
                                .padding(.vertical, 6)
                                .background(AppTheme.cardAlt)
                                .clipShape(Capsule())
                        }
                    }
                }
            }
        }
        .cardStyle()
    }

    private var debugPanel: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text("Debug")
                .font(.caption.weight(.semibold))
                .foregroundStyle(.secondary)
            Text("pipeline: \(viewModel.captureSourceLabel)")
                .font(.caption2)
                .foregroundStyle(.secondary)
            Text("observations: \(viewModel.observations.count), groups: \(viewModel.draftDetections.filter { $0.stackCount > 0 || $0.looseCount > 0 }.count)")
                .font(.caption2)
                .foregroundStyle(.secondary)
            Text("decode→nms→dedup: \(viewModel.debugStats.decodedCount) → \(viewModel.debugStats.afterNMSCount) → \(viewModel.debugStats.afterDedupCount)")
                .font(.caption2)
                .foregroundStyle(.secondary)
            if let best = viewModel.observations.map(\.confidence).max() {
                Text("max confidence: \(Int(best * 100))%")
                    .font(.caption2)
                    .foregroundStyle(.secondary)
            }
            if !rawClassSummaryText.isEmpty {
                Text("raw classes: \(rawClassSummaryText)")
                    .font(.caption2)
                    .foregroundStyle(.secondary)
            }
            if let rawBest = viewModel.observations.compactMap(\.rawModelClassConfidence).max() {
                Text("raw max confidence: \(Int(rawBest * 100))%")
                    .font(.caption2)
                    .foregroundStyle(.secondary)
            }
        }
        .padding(8)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(AppTheme.cardAlt.opacity(0.45))
        .clipShape(RoundedRectangle(cornerRadius: 10, style: .continuous))
    }

    private var groupedObservationLabels: [String] {
        let counts = Dictionary(grouping: viewModel.observations, by: \.predictedColorName)
            .mapValues { $0.count }
        return counts
            .keys
            .sorted()
            .map { "\($0.capitalized) \(counts[$0] ?? 0)" }
    }

    private var rawClassSummaryText: String {
        let labels = viewModel.observations.compactMap(\.rawModelClassLabel)
        guard !labels.isEmpty else { return "" }
        let counts = Dictionary(grouping: labels, by: { $0 }).mapValues { $0.count }
        return counts
            .keys
            .sorted()
            .map { "\($0):\(counts[$0] ?? 0)" }
            .joined(separator: ", ")
    }
}

struct DetectionReviewView: View {
    @EnvironmentObject private var appState: AppState
    @ObservedObject var viewModel: CaptureFlowViewModel

    var body: some View {
        List {
            Section("Detected Groups") {
                ForEach($viewModel.draftDetections) { $detection in
                    if let configIndex = appState.chipConfigs.firstIndex(where: { $0.id == detection.chipColorID }) {
                        DetectionEditorCard(
                            config: $appState.chipConfigs[configIndex],
                            detection: $detection
                        )
                    }
                }
            }

            Section {
                Button("Save stack snapshot") {
                    appState.applyDetections(viewModel.draftDetections)
                }
                .frame(maxWidth: .infinity)
            }

            if !appState.latestDetections.isEmpty {
                Section("Current Summary") {
                    NavigationLink {
                        StackSummaryView()
                    } label: {
                        Text("Open current stack summary")
                    }
                }
            }
        }
        .navigationTitle("Review Result")
        .navigationBarTitleDisplayMode(.inline)
    }
}

private struct ChipPreviewOverlay: View {
    let preview: Image
    let observations: [ChipTopObservation]
    let chipConfigs: [ChipColorConfig]
    let imageSize: CGSize?

    var body: some View {
        GeometryReader { proxy in
            ZStack {
                preview
                    .resizable()
                    .scaledToFit()
                    .frame(width: proxy.size.width, height: proxy.size.height)
                    .background(Color.black.opacity(0.15))
                    .overlay(Color.black.opacity(0.08))

                ForEach(observations) { observation in
                    ObservationMarker(
                        observation: observation,
                        size: proxy.size,
                        imageSize: imageSize,
                        showLabel: observations.count <= 40,
                        strokeColor: color(for: observation)
                    )
                }
            }
            .clipShape(RoundedRectangle(cornerRadius: 18, style: .continuous))
        }
    }

    private func color(for observation: ChipTopObservation) -> Color {
        if let config = chipConfigs.first(where: { $0.id == observation.predictedChipColorID }) {
            return config.swiftUIColor
        }
        return Color(hex: fallbackHex(for: observation.predictedColorName))
    }

    private func fallbackHex(for label: String) -> String {
        switch label.lowercased() {
        case "orange":
            return "#E27A3F"
        case "pink":
            return "#D98AA8"
        case "green":
            return "#3C8D68"
        case "purple":
            return "#8765C9"
        case "black":
            return "#2D2D2D"
        default:
            return "#B6B6B6"
        }
    }
}

private struct ObservationMarker: View {
    let observation: ChipTopObservation
    let size: CGSize
    let imageSize: CGSize?
    let showLabel: Bool
    let strokeColor: Color

    var body: some View {
        ZStack(alignment: .topLeading) {
            Circle()
                .stroke(strokeColor, lineWidth: 3)
                .frame(width: diameter, height: diameter)

            if showLabel {
                Text(observation.predictedColorName.capitalized)
                    .font(.caption2.weight(.bold))
                    .padding(.horizontal, 6)
                    .padding(.vertical, 4)
                    .background(strokeColor.opacity(0.92))
                    .foregroundStyle(.white)
                    .clipShape(Capsule())
                    .offset(x: -10, y: -28)
            }
        }
        .position(x: markerX, y: markerY)
    }

    private var diameter: CGFloat {
        max(22, observation.candidate.normalizedRadius * drawRect.width * 2)
    }

    private var markerX: CGFloat {
        drawRect.minX + observation.candidate.normalizedCenter.x * drawRect.width
    }

    private var markerY: CGFloat {
        drawRect.minY + observation.candidate.normalizedCenter.y * drawRect.height
    }

    private var drawRect: CGRect {
        guard let imageSize else {
            return CGRect(origin: .zero, size: size)
        }
        guard imageSize.width > 0, imageSize.height > 0, size.width > 0, size.height > 0 else {
            return CGRect(origin: .zero, size: size)
        }

        let imageAspect = imageSize.width / imageSize.height
        let viewAspect = size.width / size.height

        if imageAspect > viewAspect {
            let width = size.width
            let height = width / imageAspect
            return CGRect(x: 0, y: (size.height - height) / 2, width: width, height: height)
        } else {
            let height = size.height
            let width = height * imageAspect
            return CGRect(x: (size.width - width) / 2, y: 0, width: width, height: height)
        }
    }
}

private struct DetectionEditorCard: View {
    @Binding var config: ChipColorConfig
    @Binding var detection: StackDetectionResult

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Circle()
                    .fill(config.swiftUIColor)
                    .frame(width: 16, height: 16)
                Text(config.name)
                    .font(.headline)
                Spacer()
                Text("\(Int(detection.confidence * 100))%")
                    .foregroundStyle(.secondary)
            }
            Stepper("Stacks: \(detection.stackCount)", value: $detection.stackCount, in: 0...20)
            Stepper("Loose chips: \(detection.looseCount)", value: $detection.looseCount, in: 0...19)
            Stepper("Per stack: \(config.stackSize)", value: $config.stackSize, in: 5...40, step: 5)
            Stepper("Denomination: \(config.denomination.formattedCurrency)", value: $config.denomination, in: 25...100000, step: 25)
        }
        .padding(.vertical, 6)
    }
}

struct StackSummaryView: View {
    @EnvironmentObject private var appState: AppState

    var body: some View {
        List {
            Section("Current Totals") {
                row(title: "Stack Value", value: appState.totalChipValue.formattedCurrency)
                row(title: "Big Blinds", value: appState.currentBigBlinds.bbText)
                row(title: "Blind Level", value: "\(appState.smallBlind) / \(appState.bigBlind)")
            }

            Section("Color Breakdown") {
                ForEach(appState.latestDetections) { detection in
                    if let config = appState.chipConfigs.first(where: { $0.id == detection.chipColorID }) {
                        let totalCount = detection.stackCount * config.stackSize + detection.looseCount
                        row(
                            title: config.name,
                            value: "\(totalCount) chips / \((totalCount * config.denomination).formattedCurrency)"
                        )
                    }
                }
            }
        }
        .navigationTitle("Stack Summary")
        .navigationBarTitleDisplayMode(.inline)
    }

    private func row(title: String, value: String) -> some View {
        HStack {
            Text(title)
            Spacer()
            Text(value)
                .fontWeight(.semibold)
        }
    }
}

#Preview {
    NavigationStack {
        CaptureIntroView()
            .environmentObject(AppState())
    }
}
