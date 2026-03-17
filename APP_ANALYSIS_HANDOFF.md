# App Analysis Handoff

## 目標

讓離線 prototype 與 iOS app 使用同一個概念模型，避免之後真實分析接進 app 時又重做一輪資料格式。

## App 端目前需要的資料

每個 observation 至少要有：

- `predictedChipColorID` 或可映射到既有 chip config 的顏色名稱
- `predictedColorName`
- `normalizedCenter`
- `normalizedRadius`
- `confidence`

聚合後每個顏色卡片至少要有：

- `chipColorID`
- `stackCount`
- `looseCount`
- `confidence`

## 為什麼使用 normalized 座標

SwiftUI 預覽圖片的實際顯示尺寸會隨裝置與版面改變。

所以 overlay 不適合用原始像素座標，而應該用：

- `normalizedCenter.x` in `0...1`
- `normalizedCenter.y` in `0...1`
- `normalizedRadius` relative to image width

這樣不管圖片顯示成多大，都能把標記畫在正確位置附近。

## Prototype JSON 對接方向

離線腳本目前輸出的是像素座標，後續可加一層轉換：

- `normalized_x = center_x / image_width`
- `normalized_y = center_y / image_height`
- `normalized_radius = radius / image_width`

再餵給 app 的 `ChipTopObservation`。

## 下一步建議

1. 讓 Python prototype 額外輸出 normalized 座標
2. 定義一份可被 app 讀取的 JSON schema
3. 在 iOS app 新增 sample analysis loader
4. 再把真實 runtime pipeline 接進來
