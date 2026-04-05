import SwiftUI
import UIKit

struct HandRecorderView: View {
    @State private var selectedMode: HandRecordMode = .preflopAllIn
    @State private var activePage: RecorderPage = .record
    @State private var tableSize: Int = 7
    @State private var activePositionIDs: Set<String> = []
    @State private var selectedStreet: PokerStreet = .preflop
    @State private var selectedPositionID: String = "lj"
    @State private var selectedActor: ActionActor = .hero
    @State private var heroCards: [PlayingCard?] = [nil, nil]
    @State private var boardCards: [PlayingCard?] = [nil, nil, nil, nil, nil]
    @State private var heroStackBB: Double = 19
    @State private var stackByPositionID: [String: Double] = [:]
    @State private var oldHeroStackCache: Double = 19
    @State private var actionAmountBB: Double = 2.5
    @State private var recordedActions: [HandActionEntry] = []
    @State private var editingSlot: CardEditingSlot?
    @State private var savedHands: [SavedHandRecord] = HandRecordStorage.load()
    @State private var copiedSummaryLabel: String?
    @State private var shareItems: [Any] = []
    @State private var isShareSheetPresented = false
    @State private var stackInputTarget: StackInputTarget?
    @State private var stackInputText: String = ""
    @State private var isStackInputPresented = false

    var body: some View {
        mainContent
        .background(AppTheme.background.ignoresSafeArea())
        .navigationTitle("Hand Recorder")
        .navigationBarTitleDisplayMode(.inline)
        .safeAreaInset(edge: .bottom, spacing: 0) { bottomPickerInset }
        .sheet(isPresented: $isShareSheetPresented) {
            ActivityShareSheet(activityItems: shareItems)
        }
        .alert("Edit Stack (BB)", isPresented: $isStackInputPresented) {
            TextField("Stack BB", text: $stackInputText)
                .keyboardType(.decimalPad)
            Button("Cancel", role: .cancel) {}
            Button("Save") {
                commitStackInput()
            }
        } message: {
            Text("Enter stack size in BB (0 - 100).")
        }
        .animation(.easeInOut(duration: 0.18), value: editingSlot)
        .onAppear {
            syncTableState()
            syncActionAmountRange()
        }
        .onChange(of: tableSize) { _, _ in
            syncTableState()
        }
        .onChange(of: selectedPositionID) { _, _ in
            syncTableState()
        }
        .onChange(of: heroStackBB) { _, newValue in
            applyHeroStackToUneditedPlayers(newValue: newValue)
            syncActionAmountRange()
        }
        .onChange(of: selectedActor) { _, _ in
            syncActionAmountRange()
        }
        .onChange(of: stackByPositionID) { _, _ in
            syncActionAmountRange()
        }
        .onChange(of: recordedActions.count) { _, _ in
            syncActionAmountRange()
        }
    }

    private var mainContent: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 20) {
                Picker("Recorder Page", selection: $activePage) {
                    ForEach(RecorderPage.allCases) { page in
                        Text(page.title).tag(page)
                    }
                }
                .pickerStyle(.segmented)

                if activePage == .record {
                    introCard
                    tableCard
                    positionCard
                    actorCard
                    heroHandCard
                    actionsCard
                    timelineCard
                } else {
                    summaryCard
                    savedHandsCard
                }
            }
            .padding(20)
        }
    }

    @ViewBuilder
    private var bottomPickerInset: some View {
        if let slot = editingSlot {
            VStack(spacing: 8) {
                HStack {
                    Text(slot.title)
                        .font(.subheadline.weight(.semibold))
                        .foregroundStyle(.white.opacity(0.8))
                    Spacer()
                    Button {
                        editingSlot = nil
                    } label: {
                        Image(systemName: "xmark.circle.fill")
                            .font(.title3)
                            .foregroundStyle(.white.opacity(0.7))
                    }
                    .buttonStyle(.plain)
                }

                CardPickerSheet(
                    initialCard: card(for: slot)
                ) { card in
                    apply(card: card, to: slot)
                }
            }
            .padding(.horizontal, 12)
            .padding(.top, 8)
            .padding(.bottom, 8)
            .background(.ultraThinMaterial)
            .overlay(alignment: .top) {
                Divider()
                    .overlay(.white.opacity(0.12))
            }
        }
    }

    private var introCard: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("Fast note flow")
                .font(.title3.weight(.semibold))
            Text("Target: preflop all-in spots in 10 seconds, full postflop lines in about 30 seconds.")
                .foregroundStyle(.white.opacity(0.72))
        }
        .cardStyle()
    }

    private var modeCard: some View {
        VStack(alignment: .leading, spacing: 14) {
            Text("Record Mode")
                .font(.title3.weight(.semibold))

            Picker("Record Mode", selection: $selectedMode) {
                ForEach(HandRecordMode.allCases) { mode in
                    Text(mode.title).tag(mode)
                }
            }
            .pickerStyle(.segmented)

            Text(selectedMode.helperText)
                .font(.footnote)
                .foregroundStyle(.white.opacity(0.68))
        }
        .cardStyle()
    }

    private var tableCard: some View {
        VStack(alignment: .leading, spacing: 14) {
            HStack {
                Text("Table Size")
                    .font(.title3.weight(.semibold))
                Spacer()
                Text("\(tableSize)-max")
                    .font(.subheadline.weight(.semibold))
                    .padding(.horizontal, 12)
                    .padding(.vertical, 8)
                    .background(AppTheme.cardAlt)
                    .clipShape(Capsule())
            }

            Slider(
                value: Binding(
                    get: { Double(tableSize) },
                    set: { tableSize = Int($0.rounded()) }
                ),
                in: 2...9,
                step: 1
            )

            HStack {
                Text("2")
                Spacer()
                Text("9")
            }
            .font(.caption)
            .foregroundStyle(.white.opacity(0.62))
        }
        .cardStyle()
    }

    private var playersInHandSummary: String {
        let opponents = max(0, activePositionIDs.count - 1)
        return "\(activePositionIDs.count) players in hand (\(opponents) opponents)"
    }

    private var sortedActivePositionIDs: [String] {
        positionsOrderedFromHero
            .map(\.id)
            .filter { activePositionIDs.contains($0) }
    }

    private var playerToggleCard: some View {
        VStack(alignment: .leading, spacing: 14) {
            HStack {
                Text("Players In Hand")
                    .font(.headline)
                Spacer()
                Text(playersInHandSummary)
                    .font(.caption.weight(.semibold))
                    .foregroundStyle(.white.opacity(0.7))
            }

            ScrollView(.horizontal, showsIndicators: false) {
                HStack(spacing: 10) {
                    ForEach(positionsOrderedFromHero) { position in
                        let isHero = position.id == selectedPositionID
                        let isActive = activePositionIDs.contains(position.id)

                        Button {
                            if isHero {
                                return
                            }
                            togglePositionInHand(position.id)
                        } label: {
                            VStack(spacing: 4) {
                                Text(position.shortLabel)
                                    .font(.subheadline.weight(.semibold))
                                Text(isHero ? "Hero" : (isActive ? "In" : "Out"))
                                    .font(.caption2.weight(.bold))
                                    .foregroundStyle(.white.opacity(0.75))
                            }
                            .padding(.horizontal, 14)
                            .padding(.vertical, 10)
                            .background(
                                isHero
                                ? AppTheme.chipAccent
                                : (isActive ? AppTheme.cardAlt : AppTheme.backgroundRaised)
                            )
                            .overlay(
                                RoundedRectangle(cornerRadius: 16, style: .continuous)
                                    .stroke(
                                        isHero || isActive ? AppTheme.stroke : .white.opacity(0.2),
                                        lineWidth: 1
                                    )
                            )
                            .clipShape(RoundedRectangle(cornerRadius: 16, style: .continuous))
                        }
                    }
                }
            }
        }
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
        }
        .cardStyle()
    }

    private var positionCard: some View {
        VStack(alignment: .leading, spacing: 14) {
            Text("Hero Position")
                .font(.title3.weight(.semibold))

            ScrollView(.horizontal, showsIndicators: false) {
                HStack(spacing: 10) {
                    ForEach(allPositions) { position in
                        Button {
                            selectedPositionID = position.id
                        } label: {
                            Text(position.shortLabel)
                                .font(.subheadline.weight(.semibold))
                                .padding(.horizontal, 14)
                                .padding(.vertical, 10)
                                .background(
                                    selectedPositionID == position.id
                                    ? AppTheme.chipAccent
                                    : AppTheme.cardAlt
                                )
                                .foregroundStyle(.white)
                                .clipShape(Capsule())
                        }
                    }
                }
            }

            playerToggleCard
        }
        .cardStyle()
    }

    private var actorCard: some View {
        VStack(alignment: .leading, spacing: 14) {
            Text("Who Acts & Stacks")
                .font(.title3.weight(.semibold))

            ScrollView(.horizontal, showsIndicators: false) {
                HStack(spacing: 10) {
                    ForEach(availableActorChoices) { actor in
                        let isSelected = selectedActor == actor
                        VStack(alignment: .leading, spacing: 2) {
                            Text(actor.title)
                                .font(.subheadline.weight(.semibold))
                                .lineLimit(1)
                                .minimumScaleFactor(0.8)
                            Text(actorStackSummary(for: actor))
                                .font(.caption2.weight(.semibold))
                                .lineLimit(1)
                                .minimumScaleFactor(0.8)
                        }
                        .padding(.horizontal, 14)
                        .padding(.vertical, 10)
                        .background(isSelected ? AppTheme.chipAccent : AppTheme.cardAlt)
                        .foregroundStyle(isSelected ? Color.black.opacity(0.9) : .white)
                        .clipShape(RoundedRectangle(cornerRadius: 16, style: .continuous))
                        .onTapGesture {
                            if isSelected {
                                presentStackInput(for: actor)
                            } else {
                                selectedActor = actor
                            }
                        }
                    }
                }
            }

            Divider()
                .overlay(.white.opacity(0.15))

            stackControlsContent
        }
        .cardStyle()
    }

    private var stackControlsContent: some View {
        VStack(alignment: .leading, spacing: 14) {
            HStack {
                Text("Hero \(formattedBB(heroStackBB))")
                    .font(.subheadline.weight(.semibold))
                Spacer()
            }

            HStack(spacing: 8) {
                stepChip(label: "-5", repeatsOnLongPress: true) { heroStackBB = clampedStackValue(heroStackBB - 5) }
                stepChip(label: "-") { heroStackBB = clampedStackValue(heroStackBB - 0.5) }
                stepChip(label: "+") { heroStackBB = clampedStackValue(heroStackBB + 0.5) }
                stepChip(label: "+5", repeatsOnLongPress: true) { heroStackBB = clampedStackValue(heroStackBB + 5) }
            }
            .font(.footnote.weight(.semibold))

            ScrollView(.horizontal, showsIndicators: false) {
                HStack(spacing: 10) {
                    ForEach(activePositions) { position in
                        if position.id != selectedPositionID {
                            VStack(alignment: .leading, spacing: 8) {
                                Text("\(position.shortLabel) \(formattedBB(stackValue(for: position.id)))")
                                    .font(.caption.weight(.semibold))
                                    .foregroundStyle(.white.opacity(0.78))
                                HStack(spacing: 6) {
                                    stepChip(label: "-5", repeatsOnLongPress: true) {
                                        stackByPositionID[position.id] = clampedStackValue(stackValue(for: position.id) - 5)
                                    }
                                    stepChip(label: "-") {
                                        stackByPositionID[position.id] = clampedStackValue(stackValue(for: position.id) - 0.5)
                                    }
                                    stepChip(label: "+") {
                                        stackByPositionID[position.id] = clampedStackValue(stackValue(for: position.id) + 0.5)
                                    }
                                    stepChip(label: "+5", repeatsOnLongPress: true) {
                                        stackByPositionID[position.id] = clampedStackValue(stackValue(for: position.id) + 5)
                                    }
                                }
                            }
                            .padding(10)
                            .frame(width: 178)
                            .background(AppTheme.cardAlt)
                            .clipShape(RoundedRectangle(cornerRadius: 16, style: .continuous))
                        }
                    }
                }
            }
        }
    }

    private var actionsCard: some View {
        VStack(alignment: .leading, spacing: 14) {
            HStack {
                Text("Action Shortcuts")
                    .font(.title3.weight(.semibold))
                Spacer()
                Text(activeActorSummary)
                    .font(.footnote.weight(.semibold))
                    .foregroundStyle(AppTheme.chipAccent)
            }

            HStack {
                Text("Street")
                    .foregroundStyle(.white.opacity(0.72))
                Spacer()
                Text(selectedStreet.title)
                    .font(.headline)
            }

            Picker("Street", selection: $selectedStreet) {
                ForEach(activeStreets) { street in
                    Text(street.title).tag(street)
                }
            }
            .pickerStyle(.segmented)

            if selectedStreet != .preflop {
                VStack(alignment: .leading, spacing: 10) {
                    Text("Board for \(selectedStreet.title)")
                        .font(.headline)
                    ScrollView(.horizontal, showsIndicators: false) {
                        HStack(spacing: 10) {
                            ForEach(boardIndices(for: selectedStreet), id: \.self) { index in
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

            HStack {
                Text("Pot Size")
                    .foregroundStyle(.white.opacity(0.72))
                Spacer()
                Text(formattedBB(currentPotBB))
                    .font(.headline)
            }

            HStack(spacing: 8) {
                stepChip(label: "33%") { applyPotFraction(0.33) }
                stepChip(label: "50%") { applyPotFraction(0.5) }
                stepChip(label: "75%") { applyPotFraction(0.75) }
                stepChip(label: "100%") { applyPotFraction(1.0) }
            }
            .font(.footnote.weight(.semibold))

            HStack {
                Text("Amount")
                    .foregroundStyle(.white.opacity(0.72))
                Spacer()
                Text(formattedBB(actionAmountBB))
                    .font(.headline)
            }

            HStack(spacing: 8) {
                stepChip(label: "-5", repeatsOnLongPress: true) { actionAmountBB = clampedAmountValue(actionAmountBB - 5) }
                stepChip(label: "-") { actionAmountBB = clampedAmountValue(actionAmountBB - 0.5) }
                stepChip(label: "+") { actionAmountBB = clampedAmountValue(actionAmountBB + 0.5) }
                stepChip(label: "+5", repeatsOnLongPress: true) { actionAmountBB = clampedAmountValue(actionAmountBB + 5) }
            }
            .font(.footnote.weight(.semibold))

            VStack(alignment: .leading, spacing: 6) {
                HStack {
                    Text("Amount Slider")
                        .font(.caption.weight(.semibold))
                        .foregroundStyle(.white.opacity(0.65))
                    Spacer()
                    Text("Max \(formattedBB(actionAmountUpperBound))")
                        .font(.caption.weight(.semibold))
                        .foregroundStyle(.white.opacity(0.65))
                }
                Slider(
                    value: Binding(
                        get: { actionAmountBB },
                        set: { actionAmountBB = clampedAmountValue($0) }
                    ),
                    in: 0...max(1, actionAmountUpperBound),
                    step: 1
                )
                .disabled(actionAmountUpperBound <= 0)
            }

            Text("Pick street, adjust amount, then tap action.")
                .font(.footnote)
                .foregroundStyle(.white.opacity(0.62))

            LazyVGrid(columns: [GridItem(.flexible()), GridItem(.flexible()), GridItem(.flexible())], spacing: 10) {
                ForEach(filteredActions) { action in
                    Button {
                        addAction(action)
                    } label: {
                        VStack(spacing: 6) {
                            Text(action.title)
                                .font(.headline)
                            if let displayAmount = displayedAmount(for: action) {
                                Text(formattedBB(displayAmount))
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

            if filteredActions.isEmpty {
                Text("No valid shortcuts right now. Change actor/street or add previous action first.")
                    .font(.footnote)
                    .foregroundStyle(.white.opacity(0.6))
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
                        selectedStreet = .preflop
                        selectedActor = .hero
                        syncActionAmountRange()
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
                            Text(entry.summaryLine)
                                .font(.headline)
                            Text(entry.detailLine)
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

    private var summaryCard: some View {
        VStack(alignment: .leading, spacing: 14) {
            HStack {
                Text("Summary")
                    .font(.title3.weight(.semibold))
                Spacer()
                if let copiedSummaryLabel {
                    Text("Copied \(copiedSummaryLabel)")
                        .font(.caption.weight(.semibold))
                        .foregroundStyle(AppTheme.chipAccent)
                }
            }

            summaryBox(title: "Short", text: shortSummaryText)
            summaryBox(title: "Study Note", text: studySummaryText)
            summaryBox(title: "HH Text", text: handHistoryStyleText)

            HStack(spacing: 10) {
                Button("Copy HH") {
                    copyToPasteboard(buildPokerStarsStyleHandHistory(), label: "HH")
                }
                .buttonStyle(.bordered)

                Button("Share HH TXT") {
                    shareHandHistoryTextFile()
                }
                .buttonStyle(.bordered)

                Button("Share OHH") {
                    shareOHHFile()
                }
                .buttonStyle(.bordered)
            }

            HStack(spacing: 10) {
                Button("Copy Short") {
                    copyToPasteboard(shortSummaryText, label: "Short")
                }
                .buttonStyle(.bordered)

                Button("Open GTO Upload") {
                    openGtoWizardUpload()
                }
                .buttonStyle(.bordered)

                Button("Save Snapshot") {
                    saveCurrentHand()
                }
                .buttonStyle(.borderedProminent)
            }
        }
        .cardStyle()
    }

    private var savedHandsCard: some View {
        VStack(alignment: .leading, spacing: 14) {
            HStack {
                Text("Saved Hands")
                    .font(.title3.weight(.semibold))
                Spacer()
                if !savedHands.isEmpty {
                    Button("Clear All") {
                        savedHands.removeAll()
                        HandRecordStorage.save(savedHands)
                    }
                    .font(.footnote.weight(.semibold))
                }
            }

            if savedHands.isEmpty {
                Text("No saved hands yet. Save a snapshot to build your study library.")
                    .foregroundStyle(.white.opacity(0.62))
            } else {
                ForEach(savedHands.prefix(6)) { record in
                    VStack(alignment: .leading, spacing: 8) {
                        Text(record.title)
                            .font(.headline)
                        Text(record.shortSummary)
                            .font(.subheadline)
                            .foregroundStyle(.white.opacity(0.72))
                        HStack {
                            Text(record.savedAt.formatted(date: .abbreviated, time: .shortened))
                                .font(.caption)
                                .foregroundStyle(.white.opacity(0.55))
                            Spacer()
                            Button("Load") {
                                load(record)
                            }
                            .font(.footnote.weight(.semibold))
                        }
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
        let actorBefore = remainingStackValue(for: selectedActor)
        let resolvedAmount: Double?
        if action == .allIn {
            resolvedAmount = actorBefore
        } else if action == .call {
            resolvedAmount = min(amountToCall(for: selectedActor), actorBefore)
        } else {
            resolvedAmount = action.usesAmount ? min(actionAmountBB, actorBefore) : nil
        }
        let committed = resolvedAmount ?? 0
        let remainingStack = max(0, actorBefore - committed)
        recordedActions.append(
            HandActionEntry(
                actor: selectedActor,
                street: selectedStreet,
                heroPositionID: selectedPositionID,
                action: action,
                amountBB: resolvedAmount,
                remainingStackBB: remainingStack
            )
        )
        if isCurrentStreetComplete() {
            advanceStreetIfPossible()
        } else {
            moveToNextActor()
        }
        syncActionAmountRange()
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

    private func boardIndices(for street: PokerStreet) -> [Int] {
        switch street {
        case .preflop:
            return []
        case .flop:
            return [0, 1, 2]
        case .turn:
            return [0, 1, 2, 3]
        case .river:
            return [0, 1, 2, 3, 4]
        }
    }

    private func formattedBB(_ amount: Double) -> String {
        String(format: "%.1f BB", amount)
    }

    private func summaryBox(title: String, text: String) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            Text(title)
                .font(.caption.weight(.bold))
                .foregroundStyle(.white.opacity(0.62))
            Text(text)
                .font(.subheadline)
                .textSelection(.enabled)
        }
        .padding(12)
        .background(AppTheme.cardAlt)
        .clipShape(RoundedRectangle(cornerRadius: 16, style: .continuous))
    }

    private func exportChip(label: String) -> some View {
        Text(label)
            .font(.caption.weight(.semibold))
            .padding(.horizontal, 10)
            .padding(.vertical, 6)
            .background(AppTheme.backgroundRaised)
            .clipShape(Capsule())
    }

    private func stepChip(
        label: String,
        repeatsOnLongPress: Bool = false,
        action: @escaping () -> Void
    ) -> some View {
        StepChip(
            label: label,
            repeatsOnLongPress: repeatsOnLongPress,
            action: action
        )
    }

    private var activeStreets: [PokerStreet] {
        PokerStreet.allCases
    }

    private var activeActions: [HandActionKind] {
        HandActionKind.preflopFastActions
    }

    private var actionAmountUpperBound: Double {
        max(0, remainingStackValue(for: selectedActor))
    }

    private var positionsOrderedFromHero: [TablePositionOption] {
        guard let heroIndex = allPositions.firstIndex(where: { $0.id == selectedPositionID }) else {
            return allPositions
        }
        return Array(allPositions[heroIndex...]) + Array(allPositions[..<heroIndex])
    }

    private var allPositions: [TablePositionOption] {
        TablePositionOption.positions(for: tableSize)
    }

    private var activePositions: [TablePositionOption] {
        positionsOrderedFromHero.filter { activePositionIDs.contains($0.id) }
    }

    private var currentStreetActions: [HandActionEntry] {
        recordedActions.filter { $0.street == selectedStreet }
    }

    private var streetCommittedByActor: [ActionActor: Double] {
        Dictionary(
            grouping: currentStreetActions,
            by: \.actor
        ).mapValues { actions in
            actions.compactMap(\.amountBB).reduce(0, +)
        }
    }

    private var currentStreetMaxCommitted: Double {
        streetCommittedByActor.values.max() ?? 0
    }

    private var filteredActions: [HandActionKind] {
        activeActions.filter(isActionAllowed)
    }

    private var availableActorChoices: [ActionActor] {
        [.hero] + activePositions
            .filter { $0.id != selectedPositionID }
            .map { .position($0.id) }
    }

    private var activeActorSummary: String {
        "\(selectedActor.title) • \(formattedBB(remainingStackValue(for: selectedActor))) behind"
    }

    private func stackValue(for positionID: String) -> Double {
        stackByPositionID[positionID] ?? heroStackBB
    }

    private func configuredStackValue(for actor: ActionActor) -> Double {
        switch actor {
        case .hero:
            return heroStackBB
        case .position(let id):
            return stackValue(for: id)
        }
    }

    private func committedAmount(for actor: ActionActor) -> Double {
        recordedActions
            .filter { $0.actor == actor }
            .compactMap(\.amountBB)
            .reduce(0, +)
    }

    private func remainingStackValue(for actor: ActionActor) -> Double {
        max(0, configuredStackValue(for: actor) - committedAmount(for: actor))
    }

    private func actorStackSummary(for actor: ActionActor) -> String {
        formattedBB(configuredStackValue(for: actor))
    }

    private var currentPotBB: Double {
        recordedActions.compactMap(\.amountBB).reduce(0, +)
    }

    private func clampedStackValue(_ value: Double) -> Double {
        min(100, max(0, value))
    }

    private func clampedAmountValue(_ value: Double) -> Double {
        min(actionAmountUpperBound, max(0, value))
    }

    private func isActionAllowed(_ action: HandActionKind) -> Bool {
        let facingBet = isFacingBet(selectedActor)
        switch action {
        case .fold:
            return facingBet
        case .check:
            return !facingBet
        case .call:
            return facingBet && amountToCall(for: selectedActor) > 0
        case .raise:
            return remainingStackValue(for: selectedActor) > 0
        case .allIn:
            return remainingStackValue(for: selectedActor) > 0
        default:
            if facingBet {
                return !action.disallowedWhenFacingAggression
            }
            return !action.disallowedWhenFacingNoBet
        }
    }

    private var actorTurnOrder: [ActionActor] {
        [.hero] + activePositions
            .filter { $0.id != selectedPositionID }
            .map { .position($0.id) }
    }

    private func moveToNextActor() {
        let order = actorTurnOrder
        guard !order.isEmpty else { return }
        if let index = order.firstIndex(of: selectedActor), index + 1 < order.count {
            selectedActor = order[index + 1]
        } else {
            selectedActor = order.first ?? .hero
        }
    }

    private func isCurrentStreetComplete() -> Bool {
        let actors = actorTurnOrder.filter { !isActorFolded($0) }
        guard !actors.isEmpty else { return false }

        let maxCommitted = currentStreetMaxCommitted
        if maxCommitted <= 0.001 {
            return actors.allSatisfy { latestAction(on: selectedStreet, for: $0) != nil }
        }

        return actors.allSatisfy { actor in
            if latestAction(on: selectedStreet, for: actor)?.action == .fold {
                return true
            }
            return currentStreetCommitted(for: actor) + 0.001 >= maxCommitted
                && latestAction(on: selectedStreet, for: actor) != nil
        }
    }

    private func advanceStreetIfPossible() {
        if let nextStreet = selectedStreet.next {
            selectedStreet = nextStreet
            selectedActor = actorTurnOrder.first(where: { !isActorFolded($0) }) ?? .hero
        } else {
            selectedActor = actorTurnOrder.first(where: { !isActorFolded($0) }) ?? .hero
        }
    }

    private func syncActionAmountRange() {
        actionAmountBB = clampedAmountValue(actionAmountBB)
    }

    private func currentStreetCommitted(for actor: ActionActor) -> Double {
        streetCommittedByActor[actor] ?? 0
    }

    private func amountToCall(for actor: ActionActor) -> Double {
        max(0, currentStreetMaxCommitted - currentStreetCommitted(for: actor))
    }

    private func isFacingBet(_ actor: ActionActor) -> Bool {
        amountToCall(for: actor) > 0.001
    }

    private func displayedAmount(for action: HandActionKind) -> Double? {
        switch action {
        case .allIn:
            return remainingStackValue(for: selectedActor)
        case .call:
            return amountToCall(for: selectedActor)
        default:
            return action.usesAmount ? actionAmountBB : nil
        }
    }

    private func applyPotFraction(_ fraction: Double) {
        guard currentPotBB > 0 else {
            actionAmountBB = 0
            return
        }
        let raw = currentPotBB * fraction
        let roundedToHalf = (raw * 2).rounded() / 2
        actionAmountBB = clampedAmountValue(roundedToHalf)
    }

    private func presentStackInput(for actor: ActionActor) {
        switch actor {
        case .hero:
            stackInputTarget = .hero
            stackInputText = String(format: "%.1f", heroStackBB)
        case .position(let id):
            stackInputTarget = .position(id)
            stackInputText = String(format: "%.1f", stackValue(for: id))
        }
        isStackInputPresented = true
    }

    private func commitStackInput() {
        guard let value = Double(stackInputText) else { return }
        let clamped = clampedStackValue(value)
        switch stackInputTarget {
        case .hero:
            heroStackBB = clamped
        case .position(let id):
            stackByPositionID[id] = clamped
        case .none:
            break
        }
    }

    private func latestAction(on street: PokerStreet, for actor: ActionActor) -> HandActionEntry? {
        currentStreetActions.last(where: { $0.street == street && $0.actor == actor })
    }

    private func isActorFolded(_ actor: ActionActor) -> Bool {
        recordedActions.last(where: { $0.actor == actor })?.action == .fold
    }

    private func syncTableState() {
        let positions = allPositions
        let validIDs = Set(positions.map(\.id))

        if activePositionIDs.isEmpty {
            activePositionIDs = Set([selectedPositionID])
        } else {
            activePositionIDs = Set(activePositionIDs.filter { validIDs.contains($0) })
        }

        if !positions.contains(where: { $0.id == selectedPositionID }) {
            selectedPositionID = positions.first?.id ?? "button"
        }
        activePositionIDs.insert(selectedPositionID)
        if activePositionIDs.count < 2,
           let fallback = positionsOrderedFromHero.first(where: { $0.id != selectedPositionID })?.id {
            activePositionIDs.insert(fallback)
        }

        for position in allPositions {
            if stackByPositionID[position.id] == nil {
                stackByPositionID[position.id] = heroStackBB
            }
        }
        stackByPositionID = stackByPositionID.filter { validIDs.contains($0.key) }
        if case .position(let positionID) = selectedActor,
           (!activePositionIDs.contains(positionID) || positionID == selectedPositionID) {
            selectedActor = .hero
        }
    }

    private func togglePositionInHand(_ positionID: String) {
        guard positionID != selectedPositionID else { return }

        if activePositionIDs.contains(positionID) {
            guard activePositionIDs.count > 2 else { return }
            activePositionIDs.remove(positionID)
            if case .position(let actorPositionID) = selectedActor, actorPositionID == positionID {
                selectedActor = .hero
            }
        } else {
            activePositionIDs.insert(positionID)
        }
    }

    private func applyHeroStackToUneditedPlayers(newValue: Double) {
        for position in activePositions where position.id != selectedPositionID {
            if stackByPositionID[position.id] == nil || abs((stackByPositionID[position.id] ?? newValue) - oldHeroStackCache) < 0.001 {
                stackByPositionID[position.id] = newValue
            }
        }
        oldHeroStackCache = newValue
    }

    private var heroHandText: String {
        let first = heroCards[0].map(cardCode) ?? "?"
        let second = heroCards[1].map(cardCode) ?? "?"
        return "\(first)\(second)"
    }

    private var boardText: String {
        boardCards.compactMap { $0.map(cardCode) }.joined(separator: " ")
    }

    private var shortSummaryText: String {
        var base = "\(tableSize)-max • Hero \(TablePositionOption.label(for: selectedPositionID)) \(formattedBB(heroStackBB)) • \(heroHandText)"
        if !recordedActions.isEmpty {
            base += " • " + recordedActions.map(\.compressedLine).joined(separator: " • ")
        }
        if !boardText.isEmpty {
            base += " • Board \(boardText)"
        }
        return base
    }

    private var studySummaryText: String {
        var lines = [
            "Table: \(tableSize)-max",
            "Hero: \(TablePositionOption.label(for: selectedPositionID)) / \(formattedBB(heroStackBB)) / \(heroHandText)"
        ]
        if !boardText.isEmpty {
            lines.append("Board: \(boardText)")
        }
        if !recordedActions.isEmpty {
            lines.append("Line: " + recordedActions.map(\.compressedLine).joined(separator: " -> "))
        }
        return lines.joined(separator: "\n")
    }

    private var handHistoryStyleText: String {
        var lines = [
            "Table: \(tableSize)-max",
            "Hero Position: \(TablePositionOption.label(for: selectedPositionID))",
            "Hero Stack: \(formattedBB(heroStackBB))",
            "Hero Hand: \(heroHandText)"
        ]
        if !boardText.isEmpty {
            lines.append("Board: \(boardText)")
        }
        if recordedActions.isEmpty {
            lines.append("Actions: none")
        } else {
            lines.append("Actions:")
            lines += recordedActions.enumerated().map { index, entry in
                "\(index + 1). \(entry.street.title) - \(entry.summaryLine) - \(entry.detailLine)"
            }
        }
        return lines.joined(separator: "\n")
    }

    private func copyToPasteboard(_ text: String, label: String) {
        UIPasteboard.general.string = text
        copiedSummaryLabel = label
    }

    private func buildPokerStarsStyleHandHistory() -> String {
        let handID = Int(Date().timeIntervalSince1970)
        let timeStamp = Date().formatted(date: .numeric, time: .standard)
        var lines: [String] = []
        lines.append("PokerTableCompanion Hand #\(handID): Hold'em No Limit (\(tableSize)-max) - \(timeStamp)")
        lines.append("Table 'Recorder' \(tableSize)-max Seat #1 is the button")

        let seats = allPositions.enumerated().map { index, position in
            (index + 1, position)
        }
        for (seat, position) in seats {
            let stack = position.id == selectedPositionID ? heroStackBB : stackValue(for: position.id)
            lines.append("Seat \(seat): \(playerName(for: position.id)) (\(String(format: "%.1f", stack)) BB)")
        }

        lines.append("*** HOLE CARDS ***")
        lines.append("Dealt to Hero [\(heroCards.compactMap { $0.map(cardCode) }.joined(separator: " "))]")

        for street in PokerStreet.allCases {
            let entries = recordedActions.filter { $0.street == street }
            guard !entries.isEmpty else { continue }
            lines.append(streetHeader(for: street))
            for entry in entries {
                lines.append(formatActionLine(entry))
            }
        }

        lines.append("*** SUMMARY ***")
        lines.append("Total pot \(String(format: "%.1f", currentPotBB)) BB")
        return lines.joined(separator: "\n")
    }

    private func shareHandHistoryTextFile() {
        shareFile(
            named: "hand-history-\(Int(Date().timeIntervalSince1970)).txt",
            contents: buildPokerStarsStyleHandHistory(),
            label: "HH TXT"
        )
    }

    private func shareOHHFile() {
        let export = OHHExportHand(
            tableSize: tableSize,
            heroPositionID: selectedPositionID,
            heroStackBB: heroStackBB,
            stackByPositionID: stackByPositionID,
            heroCards: heroCards.compactMap { $0.map(cardCode) },
            boardCards: boardCards.compactMap { $0.map(cardCode) },
            actions: recordedActions.map {
                OHHExportAction(
                    street: $0.street.rawValue,
                    actorID: $0.actor.id,
                    action: $0.action.rawValue,
                    amountBB: $0.amountBB,
                    remainingStackBB: $0.remainingStackBB
                )
            }
        )

        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        guard let data = try? encoder.encode(["ohh": export]),
              let text = String(data: data, encoding: .utf8) else {
            copiedSummaryLabel = "Export Failed"
            return
        }

        shareFile(
            named: "hand-history-\(Int(Date().timeIntervalSince1970)).ohh",
            contents: text,
            label: "OHH"
        )
    }

    private func shareFile(named fileName: String, contents: String, label: String) {
        do {
            let url = FileManager.default.temporaryDirectory.appendingPathComponent(fileName)
            try contents.write(to: url, atomically: true, encoding: .utf8)
            shareItems = [url]
            isShareSheetPresented = true
            copiedSummaryLabel = "Shared \(label)"
        } catch {
            copiedSummaryLabel = "Share Failed"
        }
    }

    private func openGtoWizardUpload() {
        guard let url = URL(string: "https://app.gtowizard.com/analyze/uploads") else { return }
        UIApplication.shared.open(url)
    }

    private func playerName(for positionID: String) -> String {
        if positionID == selectedPositionID { return "Hero" }
        return "Villain-\(TablePositionOption.label(for: positionID))"
    }

    private func streetHeader(for street: PokerStreet) -> String {
        switch street {
        case .preflop:
            return "*** PRE-FLOP ***"
        case .flop:
            return "*** FLOP *** [\(boardCards[0...2].compactMap { $0.map(cardCode) }.joined(separator: " "))]"
        case .turn:
            return "*** TURN *** [\(boardCards[0...3].compactMap { $0.map(cardCode) }.joined(separator: " "))]"
        case .river:
            return "*** RIVER *** [\(boardCards[0...4].compactMap { $0.map(cardCode) }.joined(separator: " "))]"
        }
    }

    private func formatActionLine(_ entry: HandActionEntry) -> String {
        let actorName: String
        switch entry.actor {
        case .hero:
            actorName = "Hero"
        case .position(let id):
            actorName = "Villain-\(TablePositionOption.label(for: id))"
        }
        if let amount = entry.amountBB {
            if entry.action == .call {
                return "\(actorName): calls \(String(format: "%.1f", amount)) BB"
            } else if entry.action == .raise {
                return "\(actorName): raises to \(String(format: "%.1f", amount)) BB"
            } else if entry.action == .allIn {
                return "\(actorName): raises \(String(format: "%.1f", amount)) BB and is all-in"
            }
            return "\(actorName): \(entry.action.timelineVerb) \(String(format: "%.1f", amount)) BB"
        }
        return "\(actorName): \(entry.action.timelineVerb)"
    }

    private func cardCode(_ card: PlayingCard) -> String {
        "\(card.rank.rawValue)\(card.suit.code)"
    }

    private func saveCurrentHand() {
        let snapshot = SavedHandRecord(
            title: "\(tableSize)-max \(TablePositionOption.label(for: selectedPositionID)) \(heroHandText)",
            savedAt: .now,
            modeRawValue: selectedMode.rawValue,
            tableSize: tableSize,
            selectedStreetRawValue: selectedStreet.rawValue,
            selectedPositionID: selectedPositionID,
            activePositionIDs: sortedActivePositionIDs,
            heroStackBB: heroStackBB,
            stackByPositionID: stackByPositionID,
            heroCards: heroCards.map { $0.map(SavedCard.init) },
            boardCards: boardCards.map { $0.map(SavedCard.init) },
            recordedActions: recordedActions.map(SavedActionEntry.init),
            shortSummary: shortSummaryText
        )
        savedHands.insert(snapshot, at: 0)
        HandRecordStorage.save(savedHands)
    }

    private func load(_ record: SavedHandRecord) {
        selectedMode = HandRecordMode(rawValue: record.modeRawValue) ?? .preflopAllIn
        tableSize = record.tableSize
        selectedStreet = PokerStreet(rawValue: record.selectedStreetRawValue) ?? .preflop
        selectedPositionID = record.selectedPositionID
        activePositionIDs = Set(record.activePositionIDs ?? TablePositionOption.positions(for: record.tableSize).map(\.id))
        heroStackBB = record.heroStackBB
        oldHeroStackCache = record.heroStackBB
        stackByPositionID = record.stackByPositionID
        heroCards = record.heroCards.map { $0.map { PlayingCard(saved: $0) } }
        boardCards = record.boardCards.map { $0.map { PlayingCard(saved: $0) } }
        recordedActions = record.recordedActions.map { HandActionEntry(saved: $0) }
        syncTableState()
    }
}

private struct PokerCardButton: View {
    let card: PlayingCard?
    let placeholder: String
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            ZStack {
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
                    cornerGlyph
                    Spacer()
                }
                Spacer()
                Text(card.suit.symbol)
                    .font(.system(size: 28, weight: .bold))
                    .foregroundStyle(card.suit.color)
                    .minimumScaleFactor(0.7)
                Spacer()
                HStack {
                    Spacer()
                    cornerGlyph
                        .rotationEffect(.degrees(180))
                }
            }
            .padding(10)
        }
        .aspectRatio(63.0 / 88.0, contentMode: .fit)
        .clipped()
    }

    private var cornerGlyph: some View {
        VStack(alignment: .leading, spacing: 2) {
            Text(card.rank.rawValue)
                .font(.system(size: 17, weight: .bold))
                .minimumScaleFactor(0.75)
            Text(card.suit.symbol)
                .font(.system(size: 13, weight: .bold))
                .minimumScaleFactor(0.75)
        }
        .foregroundStyle(card.suit.color)
    }
}

private struct CardPickerSheet: View {
    @State private var selectedRank: CardRank
    @State private var selectedSuit: CardSuit

    let onConfirm: (PlayingCard) -> Void

    init(initialCard: PlayingCard?, onConfirm: @escaping (PlayingCard) -> Void) {
        _selectedRank = State(initialValue: initialCard?.rank ?? .ace)
        _selectedSuit = State(initialValue: initialCard?.suit ?? .spades)
        self.onConfirm = onConfirm
    }

    var body: some View {
        HStack(spacing: 0) {
            Picker("Rank", selection: $selectedRank) {
                ForEach(CardRank.pickerOrder) { rank in
                    Text(rank.pickerLabel).tag(rank)
                }
            }
            .pickerStyle(.wheel)
            .frame(maxWidth: .infinity)

            Divider()
                .overlay(.white.opacity(0.2))

            Picker("Suit", selection: $selectedSuit) {
                ForEach(CardSuit.allCases) { suit in
                    Text("\(suit.symbol) \(suit.shortLabel)").tag(suit)
                }
            }
            .pickerStyle(.wheel)
            .frame(maxWidth: .infinity)
        }
        .frame(height: 190)
        .padding(.horizontal, 12)
        .onAppear {
            commitSelection()
        }
        .onChange(of: selectedRank) { _, _ in
            commitSelection()
        }
        .onChange(of: selectedSuit) { _, _ in
            commitSelection()
        }
    }

    private func commitSelection() {
        onConfirm(PlayingCard(rank: selectedRank, suit: selectedSuit))
    }
}

private struct StepChip: View {
    let label: String
    let repeatsOnLongPress: Bool
    let action: () -> Void

    @State private var repeatTimer: Timer?

    var body: some View {
        Text(label)
            .padding(.horizontal, 10)
            .padding(.vertical, 7)
            .frame(minWidth: 38)
            .background(AppTheme.backgroundRaised)
            .foregroundStyle(.white)
            .clipShape(Capsule())
            .contentShape(Capsule())
            .onTapGesture {
                action()
            }
            .onLongPressGesture(
                minimumDuration: 0.35,
                pressing: { isPressing in
                    if !isPressing {
                        stopRepeating()
                    }
                },
                perform: {
                    guard repeatsOnLongPress else { return }
                    startRepeating()
                }
            )
            .onDisappear {
                stopRepeating()
            }
    }

    private func startRepeating() {
        stopRepeating()
        action()
        repeatTimer = Timer.scheduledTimer(withTimeInterval: 0.1, repeats: true) { _ in
            action()
        }
        if let repeatTimer {
            RunLoop.main.add(repeatTimer, forMode: .common)
        }
    }

    private func stopRepeating() {
        repeatTimer?.invalidate()
        repeatTimer = nil
    }
}

private enum CardEditingSlot: Identifiable, Equatable {
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
    let actor: ActionActor
    let street: PokerStreet
    let heroPositionID: String
    let action: HandActionKind
    let amountBB: Double?
    let remainingStackBB: Double?

    var amountText: String {
        if let amountBB {
            return String(format: "%.1f BB", amountBB)
        }
        return "No amount"
    }

    var summaryLine: String {
        "\(actor.summaryLabel(heroPositionID: heroPositionID)) • \(action.title)"
    }

    var detailLine: String {
        let base: String
        if let amountBB {
            base = "\(action.timelineVerb) \(String(format: "%.1f", amountBB)) BB"
        } else {
            base = action.timelineVerb
        }
        if let remainingStackBB {
            return "\(base) • behind \(String(format: "%.1f", remainingStackBB)) BB"
        }
        return base
    }

    var compressedLine: String {
        let actorLabel = actor.summaryLabel(heroPositionID: heroPositionID)
        if let amountBB {
            return "\(actorLabel) \(action.timelineVerb) \(String(format: "%.1f", amountBB))"
        }
        return "\(actorLabel) \(action.timelineVerb)"
    }
}

private enum HandRecordMode: String, CaseIterable, Identifiable {
    case preflopAllIn
    case fullHand

    var id: String { rawValue }

    var title: String {
        switch self {
        case .preflopAllIn: return "Preflop All-in"
        case .fullHand: return "Full Hand"
        }
    }

    var helperText: String {
        switch self {
        case .preflopAllIn:
            return "Optimized for fast preflop decisions."
        case .fullHand:
            return "Use this when the hand continues postflop and you need full street-by-street notes."
        }
    }
}

private enum RecorderPage: String, CaseIterable, Identifiable {
    case record
    case library

    var id: String { rawValue }

    var title: String {
        switch self {
        case .record: return "Record"
        case .library: return "Library"
        }
    }
}

private enum StackInputTarget: Equatable {
    case hero
    case position(String)
}

private enum ActionActor: Hashable, Identifiable {
    case hero
    case position(String)

    var id: String {
        switch self {
        case .hero:
            return "hero"
        case .position(let id):
            return "position-\(id)"
        }
    }

    var title: String {
        switch self {
        case .hero:
            return "Hero"
        case .position(let id):
            return TablePositionOption.label(for: id)
        }
    }

    var subtitle: String {
        switch self {
        case .hero:
            return "You"
        case .position:
            return "Opponent position"
        }
    }

    func summaryLabel(heroPositionID: String) -> String {
        switch self {
        case .hero:
            return "Hero (\(TablePositionOption.label(for: heroPositionID)))"
        case .position(let id):
            return TablePositionOption.label(for: id)
        }
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

    var next: PokerStreet? {
        switch self {
        case .preflop: return .flop
        case .flop: return .turn
        case .turn: return .river
        case .river: return nil
        }
    }
}

private struct TablePositionOption: Identifiable, Hashable {
    let id: String
    let shortLabel: String

    static func positions(for tableSize: Int) -> [TablePositionOption] {
        switch tableSize {
        case 2:
            return [pos("sb", "SB"), pos("bb", "BB")]
        case 3:
            return [pos("btn", "BTN"), pos("sb", "SB"), pos("bb", "BB")]
        case 4:
            return [pos("co", "CO"), pos("btn", "BTN"), pos("sb", "SB"), pos("bb", "BB")]
        case 5:
            return [pos("hj", "HJ"), pos("co", "CO"), pos("btn", "BTN"), pos("sb", "SB"), pos("bb", "BB")]
        case 6:
            return [pos("utg", "UTG"), pos("hj", "HJ"), pos("co", "CO"), pos("btn", "BTN"), pos("sb", "SB"), pos("bb", "BB")]
        case 7:
            return [pos("utg", "UTG"), pos("lj", "LJ"), pos("hj", "HJ"), pos("co", "CO"), pos("btn", "BTN"), pos("sb", "SB"), pos("bb", "BB")]
        case 8:
            return [pos("utg", "UTG"), pos("utg1", "UTG+1"), pos("lj", "LJ"), pos("hj", "HJ"), pos("co", "CO"), pos("btn", "BTN"), pos("sb", "SB"), pos("bb", "BB")]
        default:
            return [pos("utg", "UTG"), pos("utg1", "UTG+1"), pos("utg2", "UTG+2"), pos("lj", "LJ"), pos("hj", "HJ"), pos("co", "CO"), pos("btn", "BTN"), pos("sb", "SB"), pos("bb", "BB")]
        }
    }

    static func label(for id: String) -> String {
        positions(for: 9).first(where: { $0.id == id })?.shortLabel ?? id.uppercased()
    }

    private static func pos(_ id: String, _ shortLabel: String) -> TablePositionOption {
        TablePositionOption(id: id, shortLabel: shortLabel)
    }
}

private enum HandActionKind: String, CaseIterable, Identifiable {
    case fold
    case check
    case call
    case raise
    case limp
    case open
    case flat
    case threeBet
    case fourBet
    case bet
    case allIn
    case jam
    case overjam
    case isoJam

    var id: String { rawValue }

    var title: String {
        switch self {
        case .fold: return "Fold"
        case .check: return "Check"
        case .call: return "Call"
        case .raise: return "Raise"
        case .limp: return "Limp"
        case .open: return "Open"
        case .flat: return "Flat"
        case .threeBet: return "3-bet"
        case .fourBet: return "4-bet"
        case .bet: return "Bet"
        case .allIn: return "All-in"
        case .jam: return "Jam"
        case .overjam: return "Overjam"
        case .isoJam: return "Iso Jam"
        }
    }

    var usesAmount: Bool {
        switch self {
        case .fold, .check:
            return false
        case .call, .raise, .limp, .open, .flat, .threeBet, .fourBet, .bet, .allIn, .jam, .overjam, .isoJam:
            return true
        }
    }

    var startsAggression: Bool {
        switch self {
        case .raise, .open, .threeBet, .fourBet, .bet, .allIn, .jam, .overjam, .isoJam:
            return true
        case .fold, .check, .call, .limp, .flat:
            return false
        }
    }

    var disallowedWhenFacingAggression: Bool {
        switch self {
        case .check, .open, .bet, .limp:
            return true
        case .fold, .call, .raise, .flat, .threeBet, .fourBet, .allIn, .jam, .overjam, .isoJam:
            return false
        }
    }

    var disallowedWhenFacingNoBet: Bool {
        switch self {
        case .fold, .call, .flat, .threeBet, .fourBet, .overjam, .isoJam:
            return true
        case .check, .raise, .limp, .open, .bet, .allIn, .jam:
            return false
        }
    }

    var timelineVerb: String {
        switch self {
        case .fold: return "fold"
        case .check: return "check"
        case .call: return "call"
        case .raise: return "raise"
        case .limp: return "limp"
        case .open: return "open"
        case .flat: return "flat"
        case .threeBet: return "3-bet"
        case .fourBet: return "4-bet"
        case .bet: return "bet"
        case .allIn: return "all-in"
        case .jam: return "jam"
        case .overjam: return "overjam"
        case .isoJam: return "iso jam"
        }
    }

    static var preflopFastActions: [HandActionKind] {
        [.fold, .check, .call, .raise, .allIn]
    }

    static var fullHandActions: [HandActionKind] {
        [.fold, .check, .call, .open, .flat, .threeBet, .fourBet, .bet, .jam, .overjam]
    }
}

private struct PlayingCard: Equatable {
    let rank: CardRank
    let suit: CardSuit

    init(rank: CardRank, suit: CardSuit) {
        self.rank = rank
        self.suit = suit
    }

    init(saved: SavedCard) {
        self.rank = CardRank(rawValue: saved.rankRawValue) ?? .ace
        self.suit = CardSuit(rawValue: saved.suitRawValue) ?? .spades
    }
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

    var pickerLabel: String {
        switch self {
        case .ace: return "A"
        case .king: return "K"
        case .queen: return "Q"
        case .jack: return "J"
        case .ten: return "10"
        case .nine: return "9"
        case .eight: return "8"
        case .seven: return "7"
        case .six: return "6"
        case .five: return "5"
        case .four: return "4"
        case .three: return "3"
        case .two: return "2"
        }
    }

    static var pickerOrder: [CardRank] {
        [.ace, .king, .queen, .jack, .ten, .nine, .eight, .seven, .six, .five, .four, .three, .two]
    }
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

    var code: String {
        switch self {
        case .spades: return "s"
        case .hearts: return "h"
        case .diamonds: return "d"
        case .clubs: return "c"
        }
    }
}

private struct SavedHandRecord: Codable, Identifiable {
    let id: UUID
    let title: String
    let savedAt: Date
    let modeRawValue: String
    let tableSize: Int
    let selectedStreetRawValue: String
    let selectedPositionID: String
    let activePositionIDs: [String]?
    let heroStackBB: Double
    let stackByPositionID: [String: Double]
    let heroCards: [SavedCard?]
    let boardCards: [SavedCard?]
    let recordedActions: [SavedActionEntry]
    let shortSummary: String

    init(
        id: UUID = UUID(),
        title: String,
        savedAt: Date,
        modeRawValue: String,
        tableSize: Int,
        selectedStreetRawValue: String,
        selectedPositionID: String,
        activePositionIDs: [String]? = nil,
        heroStackBB: Double,
        stackByPositionID: [String: Double],
        heroCards: [SavedCard?],
        boardCards: [SavedCard?],
        recordedActions: [SavedActionEntry],
        shortSummary: String
    ) {
        self.id = id
        self.title = title
        self.savedAt = savedAt
        self.modeRawValue = modeRawValue
        self.tableSize = tableSize
        self.selectedStreetRawValue = selectedStreetRawValue
        self.selectedPositionID = selectedPositionID
        self.activePositionIDs = activePositionIDs
        self.heroStackBB = heroStackBB
        self.stackByPositionID = stackByPositionID
        self.heroCards = heroCards
        self.boardCards = boardCards
        self.recordedActions = recordedActions
        self.shortSummary = shortSummary
    }
}

private struct SavedCard: Codable {
    let rankRawValue: String
    let suitRawValue: String

    init(_ card: PlayingCard) {
        self.rankRawValue = card.rank.rawValue
        self.suitRawValue = card.suit.rawValue
    }
}

private struct SavedActionEntry: Codable {
    let actorID: String
    let streetRawValue: String
    let heroPositionID: String
    let actionRawValue: String
    let amountBB: Double?
    let remainingStackBB: Double?

    init(_ entry: HandActionEntry) {
        self.actorID = entry.actor.id
        self.streetRawValue = entry.street.rawValue
        self.heroPositionID = entry.heroPositionID
        self.actionRawValue = entry.action.rawValue
        self.amountBB = entry.amountBB
        self.remainingStackBB = entry.remainingStackBB
    }
}

private struct OHHExportHand: Codable {
    let spec_version: String = "0.1-draft"
    let game_type: String = "NLH"
    let table_size: Int
    let hero_position_id: String
    let hero_stack_bb: Double
    let stack_by_position_id: [String: Double]
    let hero_cards: [String]
    let board_cards: [String]
    let actions: [OHHExportAction]

    init(
        tableSize: Int,
        heroPositionID: String,
        heroStackBB: Double,
        stackByPositionID: [String: Double],
        heroCards: [String],
        boardCards: [String],
        actions: [OHHExportAction]
    ) {
        self.table_size = tableSize
        self.hero_position_id = heroPositionID
        self.hero_stack_bb = heroStackBB
        self.stack_by_position_id = stackByPositionID
        self.hero_cards = heroCards
        self.board_cards = boardCards
        self.actions = actions
    }
}

private struct OHHExportAction: Codable {
    let street: String
    let actor_id: String
    let action: String
    let amount_bb: Double?
    let remaining_stack_bb: Double?

    init(street: String, actorID: String, action: String, amountBB: Double?, remainingStackBB: Double?) {
        self.street = street
        self.actor_id = actorID
        self.action = action
        self.amount_bb = amountBB
        self.remaining_stack_bb = remainingStackBB
    }
}

private extension HandActionEntry {
    init(saved: SavedActionEntry) {
        self.init(
            actor: ActionActor(savedID: saved.actorID),
            street: PokerStreet(rawValue: saved.streetRawValue) ?? .preflop,
            heroPositionID: saved.heroPositionID,
            action: HandActionKind(rawValue: saved.actionRawValue) ?? .call,
            amountBB: saved.amountBB,
            remainingStackBB: saved.remainingStackBB
        )
    }
}

private extension ActionActor {
    init(savedID: String) {
        if savedID == "hero" {
            self = .hero
        } else if let positionID = savedID.split(separator: "-").last {
            self = .position(String(positionID))
        } else {
            self = .hero
        }
    }
}

private struct ActivityShareSheet: UIViewControllerRepresentable {
    let activityItems: [Any]

    func makeUIViewController(context: Context) -> UIActivityViewController {
        UIActivityViewController(activityItems: activityItems, applicationActivities: nil)
    }

    func updateUIViewController(_ uiViewController: UIActivityViewController, context: Context) {}
}

private enum HandRecordStorage {
    private static let key = "saved_hand_records_v1"

    static func load() -> [SavedHandRecord] {
        guard let data = UserDefaults.standard.data(forKey: key) else { return [] }
        return (try? JSONDecoder().decode([SavedHandRecord].self, from: data)) ?? []
    }

    static func save(_ records: [SavedHandRecord]) {
        guard let data = try? JSONEncoder().encode(records) else { return }
        UserDefaults.standard.set(data, forKey: key)
    }
}

#Preview {
    NavigationStack {
        HandRecorderView()
    }
}
