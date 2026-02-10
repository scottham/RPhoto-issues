# iOS 离线翻译能力最小测试

这个仓库只做一件事：

> **在离线状态下，验证 iOS 自带翻译能力是否可完成“语种检测 + 翻译为英文”。**

无需在线功能展示。

## 代码文件

- `IOSOfflineTranslationTest.swift`
  - `NLLanguageRecognizer`：语种检测
  - `TranslationSession`（`Translation` 框架）：翻译为英文
  - 包含 `@main` 入口，便于直接在 Xcode 跑起来

## 在 Xcode 中最简单运行方式

1. 打开 Xcode（16+），创建一个新的 **iOS App (SwiftUI)** 项目。
2. 删除默认 `ContentView.swift` 和 `App` 入口文件（或保留但不要重复 `@main`）。
3. 把仓库里的 `IOSOfflineTranslationTest.swift` 拖入项目。
4. 选择 iOS 18+ 模拟器或真机运行。

## 离线测试步骤（建议）

1. **先联网运行一次**，输入一段非英文文本并翻译（系统可能会下载语言资源）。
2. 打开飞行模式/断网。
3. 再输入同语种文本，点击“离线检测语种并翻译为英文”。

### 判定结果

- 成功显示英文翻译：离线可用（该语种资源已在本机）。
- 翻译失败：通常是离线语言资源未下载，先联网跑一次再测。
