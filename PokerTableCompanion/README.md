# PokerTableCompanion iOS Skeleton

This folder contains a SwiftUI iOS app skeleton for the poker table MVP.

## Current scope

The build is intentionally flow-first:

- Session setup
- Chip color / denomination setup
- Guided onboarding for player capture
- Photo-assisted stack update flow
- Review and manual correction of detected stack counts
- Stack value and BB summary

## Current limitations

- Image analysis is mocked by `MockChipAnalyzer`
- There is no persistence layer yet
- There is no live camera capture yet
- The Xcode project still needs a real Apple team/bundle identifier before archive/TestFlight

## Open in Xcode

Open:

- `PokerTableCompanion/PokerTableCompanion.xcodeproj`

Before running on device:

- Set your Development Team
- Update the bundle identifier if needed
- Accept the local Xcode license on this machine if Xcode prompts for it

## Suggested next step

After the flow feels right, replace `MockChipAnalyzer` with a real capture pipeline while keeping the same review screen.
