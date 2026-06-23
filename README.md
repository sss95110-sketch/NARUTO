# 🦊 NARUTO-HandSignDetection (Deep寫輪眼 - 繁體中文自訂版)

本專案是一個基於電腦視覺（YOLOX-Nano 輕量級 AI 物體檢測模型）與 ONNX Runtime 的**火影忍者手勢結印與忍術判定系統**。透過視訊鏡頭偵測您做出的手勢，並在結印順序符合配方時，以毛筆字特效發動對應的忍術！

此版本已根據您的需求進行了深度客製化，新增了多項功能與全新的鍵盤測試模式。

---

## 🚀 本次對話新增與最佳化功能

### 1. 🖼️ 遊戲封面與「START」按鈕
* 啟動 `Ninjutsu_demo.py` 時，程式不會立刻佔用鏡頭，而是先顯示 `VisualEffects/Cover.png` 作為封面。
* 封面偏下方設有一個帶有白色邊框的半透明紅色 **「START」** 按鈕。
* 用戶用滑鼠左鍵點擊「START」按鈕後，相機才會亮燈啟動並無縫進入結印偵測。

### 2. 📐 視窗自適應等比例縮放（延伸黑邊）
* 當您拖拉、拉伸或縮小 OpenCV 顯示視窗時，內部的影像**會保持等比例縮放**，絕不拉伸變形。
* 視窗中多餘的空間會自動以**黑色背景（黑邊）**填滿延伸，視覺效果非常乾淨專業。
* 封面的「START」按鈕在視窗縮放時，點擊的物理範圍也會自動進行數學換算，不論視窗拉得多大多小，點擊都能 100% 精準觸發！

### 3. 🌊「水遁・水亂波之術」特效影片疊加
* 當您在鏡頭前完成水亂波之術的結印（**辰 ➡️ 寅 ➡️ 卯**）時，系統會自動在畫面正中央播放 `VisualEffects/LQ.mov` 水流特效影片！
* 播放採用**加算混色（Additive Blending）**技術，影片中的黑色背景會自動變透明，水流特效會非常逼真地融入您的相機畫面中，播完後會自動釋放。

### 4. ⌨️ 鍵盤結印測試模式
* 新增了 **`Ninjutsu_keyboard_demo.py`** 程式。
* **免鏡頭、免 AI 模型、免載入 onnxruntime**，純粹透過鍵盤來模擬結印！點擊測試視窗後，只要敲擊鍵盤上對應手勢的字母即可結印，並觸發帥氣的毛筆字忍術。

---

## ⌨️ 鍵盤按鍵與手勢對照表

在鍵盤測試模式（`Ninjutsu_keyboard_demo.py`）中，按鍵與 14 種手勢的映射關係如下：

| 鍵盤按鍵 | 對應手勢 | 傳統地支編號 | 手勢範例圖片連結 |
| :---: | :---: | :---: | :--- |
| **q** | 子 (Rat) | 1 | [點此看手勢圖](https://user-images.githubusercontent.com/37477845/95611897-6d032d00-0a9d-11eb-86c4-de1c50c0d7b6.jpg) |
| **w** | 丑 (Ox) | 2 | [點此看手勢圖](https://user-images.githubusercontent.com/37477845/95611906-6ffe1d80-0a9d-11eb-96f6-a687b012c413.jpg) |
| **e** | 寅 (Tiger) | 3 | [點此看手勢圖](https://user-images.githubusercontent.com/37477845/95611912-712f4a80-0a9d-11eb-8cb8-fc7097e16f60.jpg) |
| **r** | 卯 (Hare) | 4 | [點此看手勢圖](https://user-images.githubusercontent.com/37477845/95611915-72607780-0a9d-11eb-9995-66524ce4f978.jpg) |
| **t** | 辰 (Dragon) | 5 | [點此看手勢圖](https://user-images.githubusercontent.com/37477845/95611920-7391a480-0a9d-11eb-8e74-db39acf90f83.jpg) |
| **y** | 巳 (Snake) | 6 | [點此看手勢圖](https://user-images.githubusercontent.com/37477845/95611922-742a3b00-0a9d-11eb-8a21-8bdf207db9bb.jpg) |
| **u** | 午 (Horse) | 7 | [點此看手勢圖](https://user-images.githubusercontent.com/37477845/95611928-755b6800-0a9d-11eb-86c0-67605ffd6e9b.jpg) |
| **i** | 未 (Ram) | 8 | [點此看手勢圖](https://user-images.githubusercontent.com/37477845/95611930-768c9500-0a9d-11eb-81c6-067b632dc43d.jpg) |
| **o** | 申 (Monkey) | 9 | [點此看手勢圖](https://user-images.githubusercontent.com/37477845/95611931-77252b80-0a9d-11eb-97d6-e3efc6f1aac3.jpg) |
| **p** | 酉 (Bird) | 10 | [點此看手勢圖](https://user-images.githubusercontent.com/37477845/95611935-77bdc200-0a9d-11eb-95e1-feb8bf7f61de.jpg) |
| **a** | 戌 (Dog) | 11 | [點此看手勢圖](https://user-images.githubusercontent.com/37477845/95611936-78eeef00-0a9d-11eb-90b3-f565e4763c50.jpg) |
| **s** | 亥 (Boar) | 12 | [點此看手勢圖](https://user-images.githubusercontent.com/37477845/95611938-7a201c00-0a9d-11eb-9d5f-1daf2405f20f.jpg) |
| **d** | 祈 / 合掌 (Gassho) | 13 | [點此看手勢圖](https://user-images.githubusercontent.com/37477845/95611943-7b514900-0a9d-11eb-97be-4fda80d17879.jpg) |
| **f** | 壬 (Mizunoe) | 14 | [點此看手勢圖](https://user-images.githubusercontent.com/37477845/95611947-7c827600-0a9d-11eb-97ae-9d7eabc58cd5.jpg) |

---

## 📜 本專案內建忍術結印對照表
*(已將結印順序中的每個手勢標上地支編號，方便您在對照時一眼看出順序)*

| 屬性/種類 | 忍術名稱 (日文) | 忍術名稱 (英文) | 結印順序 (由左至右依序比出) |
| :--- | :--- | :--- | :--- |
| **火遁** | 豪火球の術 | Fireball Jutsu | 巳 (6) ➡️ 未 (8) ➡️ 申 (9) ➡️ 亥 (12) ➡️ 午 (7) ➡️ 寅 (3) <br>*(或是：巳 (6) ➡️ 寅 (3) ➡️ 申 (9) ➡️ 亥 (12) ➡️ 午 (7) ➡️ 寅 (3))* |
| **火遁** | 鳳仙花の術 | Phoenix Flower Jutsu | 子 (1) ➡️ 寅 (3) ➡️ 戌 (11) ➡️ 丑 (2) ➡️ 卯 (4) ➡️ 寅 (3) |
| **火遁** | 龍火の術 | Dragon Flame Jutsu | 巳 (6) ➡️ 辰 (5) ➡️ 卯 (4) ➡️ 寅 (3) |
| **火遁** | 火龍炎弾の術 | Dragon Flame Bomb | 未 (8) ➡️ 午 (7) ➡️ 巳 (6) ➡️ 辰 (5) ➡️ 子 (1) ➡️ 丑 (2) ➡️ 寅 (3) |
| **水遁** | 水乱破の術 | Water Trumpet | 辰 (5) ➡️ 寅 (3) ➡️ 卯 (4) *(觸發 LQ.mov 特效)* |
| **水遁** | 水鮫弾の術 | Water Shark Bomb Jutsu | 寅 (3) ➡️ 丑 (2) ➡️ 辰 (5) ➡️ 卯 (4) ➡️ 酉 (10) ➡️ 辰 (5) ➡️ 未 (8) |
| **基本/通靈** | 分身の術 | Clone Jutsu | 未 (8) ➡️ 巳 (6) ➡️ 寅 (3) |
| **基本/通靈** | 変わり身の術 | Substitution Jutsu | 未 (8) ➡️ 亥 (12) ➡️ 丑 (2) ➡️ 戌 (11) ➡️ 巳 (6) |
| **基本/通靈** | 口寄せの術 | Summoning Jutsu | 戌 (11) ➡️ 亥 (12) ➡️ 酉 (10) ➡️ 申 (9) ➡️ 未 (8) |
| **基本/通靈** | 口寄せ 土遁追牙の術 | Summoning: Fanged Pursuit Jutsu | 寅 (3) ➡️ 巳 (6) ➡️ 辰 (5) ➡️ 戌 (11) |
| **高級禁術** | 口寄せ 穢土転生の術 | Summoning: Impure World Reincarnation | 寅 (3) ➡️ 巳 (6) ➡️ 戌 (11) ➡️ 辰 (5) ➡️ 祈 (13) |
| **高級禁術** | 屍鬼封尽の術 | Sealing Jutsu: Reaper Death Seal | 巳 (6) ➡️ 亥 (12) ➡️ 未 (8) ➡️ 卯 (4) ➡️ 戌 (11) ➡️ 子 (1) ➡️ 酉 (10) ➡️ 午 (7) ➡️ 巳 (6) ➡️ 祈 (13) |

### 🌊 終極挑戰：水遁・水龍彈之術 (44 個印)
> 丑 (2) ➡️ 申 (9) ➡️ 卯 (4) ➡️ 子 (1) ➡️ 亥 (12) ➡️ 酉 (10) ➡️ 丑 (2) ➡️ 午 (7) ➡️ 酉 (10) ➡️ 子 (1) ➡️ 寅 (3) ➡️ 戌 (11) ➡️ 寅 (3) ➡️ 巳 (6) ➡️ 丑 (2) ➡️ 未 (8) ➡️ 巳 (6) ➡️ 亥 (12) ➡️ 未 (8) ➡️ 子 (1) ➡️ 壬 (14) ➡️ 申 (9) ➡️ 酉 (10) ➡️ 辰 (5) ➡️ 酉 (10) ➡️ 丑 (2) ➡️ 午 (7) ➡️ 未 (8) ➡️ 寅 (3) ➡️ 巳 (6) ➡️ 子 (1) ➡️ 申 (9) ➡️ 卯 (4) ➡️ 亥 (12) ➡️ 辰 (5) ➡️ 未 (8) ➡️ 子 (1) ➡️ 丑 (2) ➡️ 申 (9) ➡️ 酉 (10) ➡️ 壬 (14) ➡️ 子 (1) ➡️ 亥 (12) ➡️ 酉 (10)

---

## 🛠️ 安裝與運行步驟

### 1. 安裝環境與套件
在終端機中執行以下指令以安裝依賴：
```powershell
pip install opencv-python numpy onnxruntime pillow
```

### 2. 執行相機忍術判定 Demo（推薦）
這會載入相機、顯示封面，並在您點擊 **START** 後開啟偵測：
```powershell
python Ninjutsu_demo.py
```
* **常用控制鍵**：
  * **c**：清除目前的結印歷史紀錄。
  * **ESC**：關閉畫面並離開程式。

### 3. 執行鍵盤結印測試模式
無須鏡頭，點擊視窗後按鍵盤字母即可發動忍術：
```powershell
python Ninjutsu_keyboard_demo.py
```

### 4. 執行原版簡單手勢偵測 Demo
僅標出偵測到的手勢名稱與置信度：
```powershell
python simple_demo.py
```

---

## 📄 專案目錄結構
```
├─Ninjutsu_demo.py           # 本主程式：包含封面、水亂波特效、等比例縮放、相機偵測
├─Ninjutsu_keyboard_demo.py  # 鍵盤測試模式 (新增)
├─simple_demo.py             # 簡單偵測 Demo
├─jutsu_cheatsheet.md        # 忍術招式表 (新增)
├─VisualEffects/
│  ├─Cover.png               # 封面圖
│  └─LQ.mov                  # 水亂波特效影片
├─setting/
│  ├─labels.csv              # 印的標籤檔
│  └─jutsu.csv               # 術的結印順序檔
├─model/
│  └─yolox/
│      └─yolox_nano.onnx     # AI 模型檔
└─utils/                     # 文字與 FPS 處理工具
```
#   N A R U T O  
 