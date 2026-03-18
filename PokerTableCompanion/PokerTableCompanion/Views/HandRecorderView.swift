import SwiftUI

struct HandRecorderView: View {
    @State private var selectedStreet: PokerStreet = .preflop
    @State private var selectedPosition: TablePosition = .button
    @State private var heroCards: [PlayingCard?] = [nil, nil]
    @State private var boardCards: [PlayingCard?] = [nil, nil, nil, nil, nil]
    @State private var actionAmountBB: Double = 2.5
    @State private var recordedActions: [HandActionEntry] = []
    @State private var editingSlot: CardEditingSlot?

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 20) {
                introCard
                heroHandCard
                positionCard
                streetCard
                actionsCard
                timelineCard
            }
            .padding(20)
        }
        .background(AppTheme.background.ignoresSafeArea())
        .navigationTitle("Hand Recorder")
        .navigationBarTitleDisplayMode(.inline)
        .sheet(item: $editingSlot) { slot in
            CardPickerSheet(
                title: slot.title,
                initialCard: card(for: slot)
            ) { card in
                apply(card: card, to: slot)
            }
            .presentationDetents([.medium, .large])
        }
    }

    private var introCard: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("Fast note flow")
                .font(.title3.weight(.semibold))
            Text("The target is to capture a hand line in 30 to 60 seconds. Start with position, hole cards, street, and the actions that matter.")
                .foregroundStyle(.white.opacity(0.72))
        }
        .cardStyle()
    }

    private var heroHandCard: some View {
        VStack(alignment: .leading, spacing: 14) {
            Text("Hero Cards")
                .font(.title3.weight(.semibold))

            HStack(spacing: 12) {
                ForEach(0..<2, id: \.self) { index in
                    PokerCardButton(
                        card: heroCards[index],
                        placeholder: index == 0 ? "Card 1" : "Card 2"
                    ) {
                        editingSlot = .hero(index)
                    }
                }
            }

            VStack(alignment: .leading, spacing: 10) {
                Text("Board")
                    .font(.headline)
                ScrollView(.horizontal, showsIndicators: false) {
                    HStack(spacing: 10) {
                        ForEach(0..<5, id: \.self) { index in
                            PokerCardButton(
                                card: boardCards[index],
                                placeholder: boardPlaceholder(index)
                            ) {
                                editingSlot = .board(index)
                            }
                        }
                    }
                }
            }
        }
        .cardStyle()
    }

    private var positionCard: some View {
        VStack(alignment: .leading, spacing: 14) {
            Text("Hero Position")
                .font(.title3.weight(.semibold))

            ScrollView(.horizontal, showsIndicators: false) {
                HStack(spacing: 10) {
                    ForEach(TablePosition.allCases) { position in
                        Button {
                            selectedPosition = position
                        } label: {
                            Text(position.shortLabel)
                                .font(.subheadline.weight(.semibold))
                                .padding(.horizontal, 14)
                                .padding(.vertical, 10)
                                .background(
                                    selectedPosition == position
                                    ? AppTheme.chipAccent
                                    : AppTheme.cardAlt
                                )
                                .foregroundStyle(.white)
                                .clipShape(Capsule())
                        }
                    }
                }
            }
        }
        .cardStyle()
    }

    private var streetCard: some View {
        VStack(alignment: .leading, spacing: 14) {
            Text("Street")
                .font(.title3.weight(.semibold))

            Picker("Street", selection: $selectedStreet) {
                ForEach(PokerStreet.allCases) { street in
                    Text(street.title).tag(street)
                }
            }
            .pickerStyle(.segmented)

            HStack {
                Text("Amount")
                    .foregroundStyle(.white.opacity(0.72))
                Spacer()
                Text(formattedBB(actionAmountBB))
                    .font(.headline)
            }

            Stepper(value: $actionAmountBB, in: 0...200, step: 0.5) {
                EmptyView()
            }
            .labelsHidden()
        }
        .cardStyle()
    }

    private var actionsCard: some View {
        VStack(alignment: .leading, spacing: 14) {
            Text("Quick Actions")
                .font(.title3.weight(.semibold))

            LazyVGrid(columns: [GridItem(.flexible()), GridItem(.flexible()), GridItem(.flexible())], spacing: 10) {
                ForEach(HandActionKind.allCases) { action in
                    Button {
                        addAction(action)
                    } label: {
                        VStack(spacing: 6) {
                            Text(action.title)
                                .font(.headline)
                            if action.usesAmount {
                                Text(formattedBB(actionAmountBB))
                                    .font(.caption)
                                    .foregroundStyle(.white.opacity(0.72))
                            } else {
                                Text("No amount")
                                    .font(.caption)
                                    .foregroundStyle(.white.opacity(0.5))
                            }
                        }
                        .frame(maxWidth: .infinity)
                        .padding(.vertical, 14)
                        .background(AppTheme.cardAlt)
                        .overlay(
                            RoundedRectangle(cornerRadius: 18, style: .continuous)
                                .stroke(AppTheme.stroke, lineWidth: 1)
                        )
                        .clipShape(RoundedRectangle(cornerRadius: 18, style: .continuous))
                    }
                    .buttonStyle(.plain)
                }
            }
        }
        .cardStyle()
    }

    private var timelineCard: some View {
        VStack(alignment: .leading, spacing: 14) {
            HStack {
                Text("Action Timeline")
                    .font(.title3.weight(.semibold))
                Spacer()
                if !recordedActions.isEmpty {
                    Button("Clear") {
                        recordedActions.removeAll()
                    }
                    .font(.footnote.weight(.semibold))
                }
            }

            if recordedActions.isEmpty {
                Text("No actions yet. Tap a quick action to build the line.")
                    .foregroundStyle(.white.opacity(0.62))
            } else {
                ForEach(recordedActions) { entry in
                    HStack(alignment: .top) {
                        Text(entry.street.shortLabel)
                            .font(.caption.weight(.bold))
                            .padding(.horizontal, 8)
                            .padding(.vertical, 5)
                            .background(AppTheme.chipAccent.opacity(0.18))
                            .clipShape(Capsule())
                        VStack(alignment: .leading, spacing: 4) {
                            Text("\(entry.position.shortLabel) • \(entry.action.title)")
                                .font(.headline)
                            Text(entry.amountText)
                                .font(.subheadline)
                                .foregroundStyle(.white.opacity(0.72))
                        }
                        Spacer()
                    }
                    .padding(12)
                    .background(AppTheme.cardAlt)
                    .clipShape(RoundedRectangle(cornerRadius: 16, style: .continuous))
                }
            }
        }
        .cardStyle()
    }

    private func addAction(_ action: HandActionKind) {
        recordedActions.append(
            HandActionEntry(
                street: selectedStreet,
                position: selectedPosition,
                action: action,
                amountBB: action.usesAmount ? actionAmountBB : nil
            )
        )
    }

    private func card(for slot: CardEditingSlot) -> PlayingCard? {
        switch slot {
        case .hero(let index):
            return heroCards[index]
        case .board(let index):
            return boardCards[index]
        }
    }

    private func apply(card: PlayingCard, to slot: CardEditingSlot) {
        switch slot {
        case .hero(let index):
            heroCards[index] = card
        case .board(let index):
            boardCards[index] = card
        }
    }

    private func boardPlaceholder(_ index: Int) -> String {
        switch index {
        case 0: return "Flop 1"
        case 1: return "Flop 2"
        case 2: return "Flop 3"
        case 3: return "Turn"
        default: return "River"
        }
    }

    private func formattedBB(_ amount: Double) -> String {
        String(format: "%.1f BB", amount)
    }
}

private struct PokerCardButton: View {
    let card: PlayingCard?
    let placeholder: String
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            VStack {
                if let card {
                    ModernPokerCardView(card: card)
                } else {
                    RoundedRectangle(cornerRadius: 16, style: .continuous)
                        .fill(AppTheme.backgroundRaised)
                        .overlay(
                            VStack(spacing: 6) {
                                Image(systemName: "plus.circle.fill")
                                    .font(.title2)
                                Text(placeholder)
                                    .font(.caption.weight(.semibold))
                            }
                            .foregroundStyle(.white.opacity(0.7))
                        )
                }
            }
            .frame(width: 82, height: 116)
        }
        .buttonStyle(.plain)
    }
}

private struct ModernPokerCardView: View {
    let card: PlayingCard

    var body: some View {
        ZStack {
            RoundedRectangle(cornerRadius: 16, style: .continuous)
                .fill(Color.white)
            RoundedRectangle(cornerRadius: 16, style: .continuous)
                .stroke(Color.black.opacity(0.08), lineWidth: 1)

            VStack {
                HStack(alignment: .top) {
                    VStack(alignment: .leading, spacing: 2) {
                        Text(card.rank.rawValue)
                            .font(.title3.weight(.bold))
                        Text(card.suit.symbol)
                            .font(.headline.weight(.bold))
                    }
                    .foregroundStyle(card.suit.color)
                    Spacer()
                }

                Spacer()

                Text(card.suit.symbol)
                    .font(.system(size: 30, weight: .bold))
                    .foregroundStyle(card.suit.color)

                Spacer()

                HStack {
                    Spacer()
                    VStack(alignment: .trailing, spacing: 2) {
                        Text(card.rank.rawValue)
                            .font(.title3.weight(.bold))
                        Text(card.suit.symbol)
                            .font(.headline.weight(.bold))
                    }
                    .foregroundStyle(card.suit.color)
                    .rotationEffect(.degrees(180))
                }
            }
            .padding(10)
        }
    }
}

private struct CardPickerSheet: View {
    @Environment(\.dismiss) private var dismiss
    @State private var selectedRank: CardRank
    @State private var selectedSuit: CardSuit

    let title: String
    let onConfirm: (PlayingCard) -> Void

    init(title: String, initialCard: PlayingCard?, onConfirm: @escaping (PlayingCard) -> Void) {
        _selectedRank = State(initialValue: initialCard?.rank ?? .ace)
        _selectedSuit = State(initialValue: initialCard?.suit ?? .spades)
        self.title = title
        self.onConfirm = onConfirm
    }

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(alignment: .leading, spacing: 20) {
                    ModernPokerCardView(card: PlayingCard(rank: selectedRank, suit: selectedSuit))
                        .frame(width: 120, height: 168)

                    VStack(alignment: .leading, spacing: 12) {
                        Text("Rank")
                            .font(.headline)
                        LazyVGrid(columns: Array(repeating: GridItem(.flexible()), count: 4), spacing: 10) {
                            ForEach(CardRank.allCases) { rank in
                                pickerChip(
                                    label: rank.rawValue,
                                    isSelected: selectedRank == rank
                                ) {
                                    selectedRank = rank
                                }
                            }
                        }
                    }

                    VStack(alignment: .leading, spacing: 12) {
                        Text("Suit")
                            .font(.headline)
                        HStack(spacing: 10) {
                            ForEach(CardSuit.allCases) { suit in
                                Button {
                                    selectedSuit = suit
                                } label: {
                                    VStack(spacing: 6) {
                                        Text(suit.symbol)
                                            .font(.title2.weight(.bold))
                                        Text(suit.shortLabel)
                                            .font(.caption2.weight(.semibold))
                                    }
                                    .frame(maxWidth: .infinity)
                                    .padding(.vertical, 12)
                                    .background(selectedSuit == suit ? AppTheme.chipAccent.opacity(0.18) : AppTheme.cardAlt)
                                    .overlay(
                                        RoundedRectangle(cornerRadius: 14, style: .continuous)
                                            .stroke(selectedSuit == suit ? AppTheme.chipAccent : AppTheme.stroke, lineWidth: 1)
                                    )
                                    .clipShape(RoundedRectangle(cornerRadius: 14, style: .continuous))
                                    .foregroundStyle(suit.color)
                                }
                            }
                        }
                    }
                }
                .padding(20)
            }
            .background(AppTheme.background.ignoresSafeArea())
            .navigationTitle(title)
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    Button("Save") {
                        onConfirm(PlayingCard(rank: selectedRank, suit: selectedSuit))
                        dismiss()
                    }
                }
            }
        }
    }

    private func pickerChip(label: String, isSelected: Bool, action: @escaping () -> Void) -> some View {
        Button(action: action) {
            Text(label)
                .font(.headline)
                .frame(maxWidth: .infinity)
                .padding(.vertical, 12)
                .background(isSelected ? AppTheme.chipAccent : AppTheme.cardAlt)
                .foregroundStyle(.white)
                .clipShape(RoundedRectangle(cornerRadius: 14, style: .continuous))
        }
    }
}

private enum CardEditingSlot: Identifiable {
    case hero(Int)
    case board(Int)

    var id: String {
        switch self {
        case .hero(let index):
            return "hero-\(index)"
        case .board(let index):
            return "board-\(index)"
        }
    }

    var title: String {
        switch self {
        case .hero(let index):
            return index == 0 ? "Hero Card 1" : "Hero Card 2"
        case .board(let index):
            switch index {
            case 0: return "Flop Card 1"
            case 1: return "Flop Card 2"
            case 2: return "Flop Card 3"
            case 3: return "Turn"
            default: return "River"
            }
        }
    }
}

private struct HandActionEntry: Identifiable {
    let id = UUID()
    let street: PokerStreet
    let position: TablePosition
    let action: HandActionKind
    let amountBB: Double?

    var amountText: String {
        if let amountBB {
            return String(format: "%.1f BB", amountBB)
        }
        return "No amount"
    }
}

private enum PokerStreet: String, CaseIterable, Identifiable {
    case preflop
    case flop
    case turn
    case river

    var id: String { rawValue }

    var title: String {
        switch self {
        case .preflop: return "Pre"
        case .flop: return "Flop"
        case .turn: return "Turn"
        case .river: return "River"
        }
    }

    var shortLabel: String {
        switch self {
        case .preflop: return "P"
        case .flop: return "F"
        case .turn: return "T"
        case .river: return "R"
        }
    }
}

private enum TablePosition: String, CaseIterable, Identifiable {
    case utg
    case hijack
    case cutoff
    case button
    case smallBlind
    case bigBlind

    var id: String { rawValue }

    var shortLabel: String {
        switch self {
        case .utg: return "UTG"
        case .hijack: return "HJ"
        case .cutoff: return "CO"
        case .button: return "BTN"
        case .smallBlind: return "SB"
        case .bigBlind: return "BB"
        }
    }
}

private enum HandActionKind: String, CaseIterable, Identifiable {
    case fold
    case check
    case call
    case bet
    case raise
    case allIn

    var id: String { rawValue }

    var title: String {
        switch self {
        case .fold: return "Fold"
        case .check: return "Check"
        case .call: return "Call"
        case .bet: return "Bet"
        case .raise: return "Raise"
        case .allIn: return "All-in"
        }
    }

    var usesAmount: Bool {
        switch self {
        case .fold, .check:
            return false
        case .call, .bet, .raise, .allIn:
            return true
        }
    }
}

private struct PlayingCard: Equatable {
    let rank: CardRank
    let suit: CardSuit
}

private enum CardRank: String, CaseIterable, Identifiable {
    case ace = "A"
    case king = "K"
    case queen = "Q"
    case jack = "J"
    case ten = "T"
    case nine = "9"
    case eight = "8"
    case seven = "7"
    case six = "6"
    case five = "5"
    case four = "4"
    case three = "3"
    case two = "2"

    var id: String { rawValue }
}

private enum CardSuit: String, CaseIterable, Identifiable {
    case spades
    case hearts
    case diamonds
    case clubs

    var id: String { rawValue }

    var symbol: String {
        switch self {
        case .spades: return "♠"
        case .hearts: return "♥"
        case .diamonds: return "♦"
        case .clubs: return "♣"
        }
    }

    var shortLabel: String {
        switch self {
        case .spades: return "Spade"
        case .hearts: return "Heart"
        case .diamonds: return "Diamond"
        case .clubs: return "Club"
        }
    }

    var color: Color {
        switch self {
        case .hearts, .diamonds:
            return Color(hex: "#E05A5A")
        case .spades, .clubs:
            return Color(hex: "#1A1C20")
        }
    }
}

#Preview {
    NavigationStack {
        HandRecorderView()
    }
}
