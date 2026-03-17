import Foundation
import SwiftUI

struct ChipColorConfig: Identifiable, Hashable, Codable {
    let id: UUID
    var name: String
    var denomination: Int
    var stackSize: Int
    var colorHex: String

    init(
        id: UUID = UUID(),
        name: String,
        denomination: Int,
        stackSize: Int = 20,
        colorHex: String
    ) {
        self.id = id
        self.name = name
        self.denomination = denomination
        self.stackSize = stackSize
        self.colorHex = colorHex
    }

    var swiftUIColor: Color {
        Color(hex: colorHex)
    }
}

struct StackDetectionResult: Identifiable, Hashable {
    let id: UUID
    var chipColorID: UUID
    var stackCount: Int
    var looseCount: Int
    var confidence: Double

    init(
        id: UUID = UUID(),
        chipColorID: UUID,
        stackCount: Int,
        looseCount: Int = 0,
        confidence: Double
    ) {
        self.id = id
        self.chipColorID = chipColorID
        self.stackCount = stackCount
        self.looseCount = looseCount
        self.confidence = confidence
    }
}

struct SessionSummary {
    var totalChipsValue: Int
    var bigBlinds: Double
}
