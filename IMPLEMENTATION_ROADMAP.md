# Photo To Color-Tagging Roadmap

## 目標

把 MVP 拆成一條可以持續推進、不會卡死在單一大問題上的實作路線。

最終目標不是一步到位全自動數籌碼，而是先做到：

- 匯入或拍攝照片
- 自動標出可見籌碼頂面的顏色
- 依顏色聚合成可修正的結果
- 在 app 裡讓玩家快速修正面額 / 每疊數量 / 疊數
- 算出總碼量與 BB

## Phase 1: 影像原型

輸入：

- 你現在提供的籌碼照片集

輸出：

- 每張圖的 debug overlay
- 每張圖的 JSON 結果
- 一份初步可用的顏色標記 heuristic

任務：

- 建立離線分析腳本
- 找可見籌碼頂面
- 用外圈環帶取色
- 做顏色分類
- 產出 observation 與按顏色聚合的 summary

完成條件：

- 至少在你自己的主要 chip set 上，能大致分出常用顏色

## Phase 2: Heuristic 校正

目標：

- 針對你的照片集調整參數

任務：

- 比對俯視與斜角照片表現
- 校正 HoughCircles / contour detection 參數
- 校正顏色閾值與參考色
- 觀察哪些背景或光線最容易誤判

完成條件：

- 你自己的照片在常見情況下有可接受的標色效果

## Phase 3: App Integration

目標：

- 讓 iOS app 能吃 analysis result

任務：

- 把 observation 資料畫成照片 overlay
- 把顏色聚合結果回填到底部卡片
- 玩家可以直接修正疊數 / 零頭 / 面額 / 每疊數量

完成條件：

- 從拍照到修正到 summary 可以在單次流程完成

## Phase 4: Camera-First UX

目標：

- 將流程從 demo 提升到實機可測

任務：

- 加入拍照時的 framing 指引
- 限制拍攝角度或提示使用者重新拍
- 根據清晰度與曝光判斷是否需要重拍

完成條件：

- 在 iPhone 上有穩定可重複的拍攝體驗

## Phase 5: 更進一步的數量估算

這一階段才碰：

- 疊高估算
- 更完整的 stack count heuristic
- OCR 讀面額
- 支援更多款式籌碼

## 當前最務實的成功標準

先做到：

- 看得到可見頂面
- 標得出顏色
- UI 能直接修正

只要這三點成立，MVP 就已經很有價值。
