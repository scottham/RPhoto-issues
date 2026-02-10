import SwiftUI
import NaturalLanguage
import Translation

/// Drop this view into an iOS app project (iOS 18+) to quickly verify:
/// offline language detection + translation to English.
struct IOSOfflineTranslationTestView: View {
    @State private var inputText: String = ""
    @State private var detectedLanguageCode: String = "-"
    @State private var translatedText: String = ""
    @State private var statusMessage: String = "Turn off network first (Airplane Mode), then enter text and tap the button."
    @State private var configuration: TranslationSession.Configuration?

    var body: some View {
        NavigationStack {
            VStack(alignment: .leading, spacing: 16) {
                Text("Offline Translation Capability Test")
                    .font(.title2.bold())

                Text("Before testing, manually disable network (Airplane Mode).")
                    .foregroundStyle(.orange)

                TextEditor(text: $inputText)
                    .frame(minHeight: 120)
                    .padding(8)
                    .overlay(
                        RoundedRectangle(cornerRadius: 10)
                            .stroke(.gray.opacity(0.4), lineWidth: 1)
                    )

                Button("Detect Language and Translate to English (Offline)") {
                    let trimmed = inputText.trimmingCharacters(in: .whitespacesAndNewlines)
                    guard !trimmed.isEmpty else {
                        statusMessage = "Input is empty. Please enter some text first."
                        translatedText = ""
                        detectedLanguageCode = "-"
                        return
                    }

                    guard let source = detectLanguage(from: trimmed) else {
                        statusMessage = "Language detection failed."
                        translatedText = ""
                        detectedLanguageCode = "Unrecognized"
                        return
                    }

                    detectedLanguageCode = source.languageCode?.identifier ?? source.identifier
                    statusMessage = "Language detected. Translating..."
                    configuration = TranslationSession.Configuration(source: source, target: .english)
                }
                .buttonStyle(.borderedProminent)

                Group {
                    Text("Detected language: \(detectedLanguageCode)")
                    Text("English output: \(translatedText)")
                    Text("Status: \(statusMessage)")
                        .foregroundStyle(.secondary)
                }

                Spacer()
            }
            .padding()
            .navigationTitle("Offline Translation Test")
        }
        .translationTask(configuration) { session in
            do {
                let response = try await session.translate(inputText)
                translatedText = response.targetText
                statusMessage = "Translation succeeded. If you are offline now, the test passed."
            } catch {
                translatedText = ""
                statusMessage = "Translation failed: \(error.localizedDescription)\nConnect to the internet once to download offline language resources, then retry offline."
            }
        }
    }

    private func detectLanguage(from text: String) -> Locale.Language? {
        let recognizer = NLLanguageRecognizer()
        recognizer.processString(text)
        guard let language = recognizer.dominantLanguage else { return nil }
        return Locale.Language(identifier: language.rawValue)
    }
}

/// Optional: if this is a brand-new SwiftUI app,
/// you can use this as the app entry directly.
@main
struct IOSOfflineTranslationTestApp: App {
    var body: some Scene {
        WindowGroup {
            IOSOfflineTranslationTestView()
        }
    }
}
