# Minimal iOS Offline Translation Test

This repository focuses on one single goal:

> **Verify whether iOS built-in translation can complete language detection + translation to English while offline.**

No online feature flow is included.

## File

- `IOSOfflineTranslationTest.swift`
  - `NLLanguageRecognizer`: detects source language
  - `TranslationSession` (`Translation` framework): translates to English
  - Includes a minimal `@main` app entry so it is easy to run in Xcode

## Easiest Way to Run in Xcode

1. Open Xcode (16+), create a new **iOS App (SwiftUI)** project.
2. Remove default `ContentView.swift` and default `App` entry file (or keep them, but avoid duplicate `@main`).
3. Drag `IOSOfflineTranslationTest.swift` from this repo into your project.
4. Run on an iOS 18+ simulator or a real device.

## Suggested Offline Test Steps

1. **Run once while online** and translate a non-English sentence (the system may download language resources).
2. Turn on Airplane Mode / disconnect network.
3. Enter text in the same language again and tap the translation button.

## How to Judge Result

- If English output appears: offline test passed (language resources are available locally).
- If translation fails: usually offline resources are not downloaded yet; run once online, then test again offline.
