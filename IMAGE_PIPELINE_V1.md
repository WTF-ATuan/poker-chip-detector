# Image Pipeline V1

## 目標

第一版影像流程只解這件事情：

- 從單張照片中找出可見的籌碼頂面
- 估計每個頂面屬於哪一個顏色群組
- 把顏色群組整理成可供 UI 修正的結果

第一版不解：

- 精準估算每一疊實際高度
- 完整追蹤被遮擋住的籌碼數量
- 用 OCR 穩定讀出所有面額文字
- 即時影片追蹤

## 為什麼這條路比較適合 MVP

對 MVP 而言，`自動標出顏色` 比 `自動算出整疊有幾顆` 容易很多。

使用者只要接受：

- 系統先幫忙找到可見的籌碼頂面
- 系統先猜顏色
- 使用者再修正疊數 / 零頭 / 面額

就已經能把手動成本降很多。

## 輸入假設

建議第一版先限制拍攝條件：

- 以 iPhone 俯視拍照為主
- 同色籌碼盡量放在一起
- 每個顏色不要混疊在一起
- 頂面盡量清楚可見
- 背景盡量固定在牌桌墊
- 只支援少數固定顏色，例如橘 / 粉 / 綠 / 黑 / 白

## 整體流程

### Stage 1: 預處理

輸入：

- 原始照片

處理：

- 轉成較適合分析的尺寸，例如最長邊 1280
- 做輕量去噪與對比調整
- 視需要做白平衡或亮度正規化

輸出：

- 標準化後的分析圖

###+ 建議技術

- Core Image
- Vision
- OpenCV

## Stage 2: 找可見頂面候選區

目標：

- 找出畫面中「看起來像籌碼頂面」的圓形或橢圓形區域

做法方向 A：

- Hough Circle / ellipse detection

做法方向 B：

- Vision contour detection
- 從 contour 過濾近似圓形區域

做法方向 C：

- 後續若需要再換成小型 object detector，只偵測 top chip

輸出：

- 一組候選 `ChipTopCandidate`
- 每個候選包含：
  - 中心點
  - 半徑或 bounding box
  - 基礎信心分數

## Stage 3: 候選過濾

目標：

- 把明顯不是籌碼頂面的候選移除

可用規則：

- 面積過小或過大就丟掉
- 長寬比偏離太多就丟掉
- 與其他候選高度重疊時只保留較佳者
- 位於畫面極邊緣且形狀殘缺太多者先丟掉

輸出：

- 過濾後的 `ChipTopCandidate`

## Stage 4: 顏色取樣

目標：

- 針對每個籌碼頂面取一個較穩定的代表色

關鍵原則：

- 不要直接取整顆頂面平均色
- 要避開中央文字區、白色區塊、反光區

建議作法：

- 只取距離中心某個半徑範圍的環狀區域
- 例如取 45% 到 75% 半徑之間的像素
- 在 HSV 或 Lab 空間中計算主色

因為很多 poker chip 的真正代表色是在外圈環帶，不在中心。

輸出：

- 每個候選區的代表色特徵

## Stage 5: 顏色分類

目標：

- 把每個候選區分類到既有 chip color config

第一版最推薦的方式：

- 使用者先定義 chip colors
- 系統把候選區代表色與每個 config 的參考色做距離比較
- 選最近的顏色作為預測結果

建議色彩空間：

- Lab 優先
- HSV 可當輔助

原因：

- Lab 更適合做感知色差

輸出：

- `ChipTopObservation`
- 每個 observation 包含：
  - 位置
  - 預測顏色 id
  - 顏色信心值

## Stage 6: 依顏色聚合

目標：

- 把多個頂面 observation 整理成 UI 可修正的顏色卡片

第一版先不要硬算疊高，只做：

- 每個顏色看到了多少個可見頂面
- 給一個建議 stack count 初值

可用的初始規則：

- 若同色 observation 散開，先視為多疊
- 若緊密靠在一起且部分被遮擋，先估較少疊
- MVP 可先直接把 `visibleTopCount` 當成預設疊數初值

輸出：

- `StackDetectionResult`
- `visibleTopCount`
- `detectedColorLabel`

## 第一版推薦的實作策略

### 方案 1: 全規則式 CV

流程：

- 找圓 / 橢圓
- 取環狀區顏色
- 做顏色距離分類

優點：

- 最快可以動
- 資料需求最低
- 很適合你現在的 MVP

缺點：

- 拍攝條件太亂時準確度會掉

### 方案 2: top-chip detector + 規則式顏色分類

流程：

- detector 只負責找頂面
- 顏色仍用環狀區特徵分類

優點：

- 比直接做多分類 detector 更穩

缺點：

- 要先做資料標註

### 方案 3: detector 直接分類顏色

流程：

- 直接把 top chip 分成 orange / pink / green / black...

優點：

- pipeline 比較短

缺點：

- 對資料量與拍攝變化更敏感
- 之後 chip set 客製化比較麻煩

## MVP 我最推薦的版本

先做：

1. 影像縮放與正規化
2. 找可見圓形 / 橢圓頂面
3. 對每個頂面做環狀區取色
4. 與已知 chip color config 比色
5. 回填為顏色標記與預設疊數

不要先做：

1. OCR 讀面額
2. LiDAR 高度估算
3. 整疊 3D 建模
4. 即時連續影片追蹤

## UI 對 pipeline 的需求

這個 pipeline 最適合搭配單頁式介面：

- 上半部顯示照片與顏色標記
- 下半部顯示顏色卡片

每張卡片至少需要：

- 顏色名稱
- 系統預測看到幾個頂面
- 面額
- 每疊數量
- 疊數
- 零頭

## 開發切法

### Milestone 1

- 先用 mock image observations
- 先把 overlay 資料結構與 UI 接好

### Milestone 2

- 實作簡單 circle / ellipse candidate detection
- 在測試照片上畫出 candidate overlay

### Milestone 3

- 實作環狀區取色
- 將候選區自動分類為橘 / 粉 / 綠

### Milestone 4

- 將 observation 聚合回現有 `StackDetectionResult`
- 接到目前 capture flow

## 一句話結論

第一版最值得做的不是「算出一整疊幾顆」，而是「找到可見頂面並先把顏色標對」，因為這能最快把手動工作量降下來。
