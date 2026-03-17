import Foundation

protocol ChipAnalyzing {
    func analyze(request: ChipAnalysisRequest) async -> ChipAnalysisResult
}
