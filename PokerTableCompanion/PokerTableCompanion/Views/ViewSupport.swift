import SwiftUI

enum AppTheme {
    static let background = Color(hex: "#071611")
    static let backgroundRaised = Color(hex: "#0E221B")
    static let card = Color(hex: "#112A22")
    static let cardAlt = Color(hex: "#18382E")
    static let stroke = Color.white.opacity(0.08)
    static let chipAccent = Color(hex: "#E05F34")
}

extension View {
    func cardStyle() -> some View {
        padding(18)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(AppTheme.card)
            .overlay(
                RoundedRectangle(cornerRadius: 22, style: .continuous)
                    .stroke(AppTheme.stroke, lineWidth: 1)
            )
            .clipShape(RoundedRectangle(cornerRadius: 22, style: .continuous))
    }
}

extension Int {
    var formattedCurrency: String {
        NumberFormatter.currencyFormatter.string(from: NSNumber(value: self)) ?? "\(self)"
    }
}

extension Double {
    var bbText: String {
        String(format: "%.1f BB", self)
    }
}

extension NumberFormatter {
    static let currencyFormatter: NumberFormatter = {
        let formatter = NumberFormatter()
        formatter.numberStyle = .decimal
        return formatter
    }()
}

extension Color {
    init(hex: String) {
        let hexString = hex.trimmingCharacters(in: CharacterSet.alphanumerics.inverted)
        var int: UInt64 = 0
        Scanner(string: hexString).scanHexInt64(&int)

        let a, r, g, b: UInt64
        switch hexString.count {
        case 8:
            (a, r, g, b) = (int >> 24, int >> 16 & 0xFF, int >> 8 & 0xFF, int & 0xFF)
        case 6:
            (a, r, g, b) = (255, int >> 16, int >> 8 & 0xFF, int & 0xFF)
        default:
            (a, r, g, b) = (255, 0, 0, 0)
        }

        self.init(
            .sRGB,
            red: Double(r) / 255,
            green: Double(g) / 255,
            blue: Double(b) / 255,
            opacity: Double(a) / 255
        )
    }
}
