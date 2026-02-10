import SwiftUI
import NaturalLanguage
import Translation

/// 这个 View 可以直接复制到 Xcode 的 iOS App 项目里运行（iOS 18+）。
/// 用途：快速验证“离线状态下”是否能完成语种检测 + 翻译英文。
struct IOSOfflineTranslationTestView: View {
    @State private var inputText: String = ""
    @State private var detectedLanguageCode: String = "-"
    @State private var translatedText: String = ""
    @State private var statusMessage: String = "请先断网（飞行模式），输入文本后点击按钮"
    @State private var configuration: TranslationSession.Configuration?

    var body: some View {
        NavigationStack {
            VStack(alignment: .leading, spacing: 16) {
                Text("离线翻译能力测试")
                    .font(.title2.bold())

                Text("测试前请手动关闭网络（飞行模式）")
                    .foregroundStyle(.orange)

                TextEditor(text: $inputText)
                    .frame(minHeight: 120)
                    .padding(8)
                    .overlay(
                        RoundedRectangle(cornerRadius: 10)
                            .stroke(.gray.opacity(0.4), lineWidth: 1)
                    )

                Button("离线检测语种并翻译为英文") {
                    let trimmed = inputText.trimmingCharacters(in: .whitespacesAndNewlines)
                    guard !trimmed.isEmpty else {
                        statusMessage = "输入为空，请先输入文本"
                        translatedText = ""
                        detectedLanguageCode = "-"
                        return
                    }

                    guard let source = detectLanguage(from: trimmed) else {
                        statusMessage = "语种检测失败"
                        translatedText = ""
                        detectedLanguageCode = "无法识别"
                        return
                    }

                    detectedLanguageCode = source.languageCode?.identifier ?? source.identifier
                    statusMessage = "检测完成，翻译中..."
                    configuration = TranslationSession.Configuration(source: source, target: .english)
                }
                .buttonStyle(.borderedProminent)

                Group {
                    Text("检测语种：\(detectedLanguageCode)")
                    Text("英文结果：\(translatedText)")
                    Text("状态：\(statusMessage)")
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
                statusMessage = "翻译成功（若当前为离线，即验证通过）"
            } catch {
                translatedText = ""
                statusMessage = "翻译失败：\(error.localizedDescription)\n请先联网下载该语种离线资源后再断网测试。"
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

/// 可选：如果你新建了一个空白 SwiftUI App，
/// 可以直接把下面这段当作入口（或拷贝 IOSOfflineTranslationTestView 到 ContentView 使用）。
@main
struct IOSOfflineTranslationTestApp: App {
    var body: some Scene {
        WindowGroup {
            IOSOfflineTranslationTestView()
        }
    }
}
