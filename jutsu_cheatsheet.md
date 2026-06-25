# 🦊 火影忍者結印忍術招式表 (Jutsu Cheatsheet)

本招式表完全根據本專案的設定檔 [setting/jutsu.csv](file:///C:/Users/sss95/OneDrive/Desktop/Class%20Materials/114-2/opencv/NinjaHand/NARUTO-HandSignDetection-main/setting/jutsu.csv) 與 [setting/labels.csv](file:///C:/Users/sss95/OneDrive/Desktop/Class%20Materials/114-2/opencv/NinjaHand/NARUTO-HandSignDetection-main/setting/labels.csv) 整理而成。您可以一邊運行 `Ninjutsu_demo.py` 或 `Ninjutsu_keyboard_demo.py`，一邊對照此表進行結印。

---

## 🎆 專屬特效忍術對照表 (配有動態實時特效與測試鍵)

以下招式在辨識成功後，會在畫面上觸發獨特的**半透明融合、綠幕去背、粒子膨脹或 GIF 特效**，並支援使用**鍵盤捷徑鍵**直接手動測試：

| 忍術名稱 (日文) | 屬性/種類 | 結印順序 (由左至右比出) | 測試鍵 | 視覺特效描述 |
| :--- | :--- | :--- | :---: | :--- |
| **分身の術** | 基本忍術 | 未 (8) ➡️ 巳 (6) ➡️ 寅 (3) | `Space` / `S` | 兩旁爆開大面積白煙粒子後，出現兩具完美半透明羽化融合的影分身。 |
| **千鳥** | 雷遁 / 秘術 | 丑 (2) ➡️ 卯 (4) ➡️ 申 (9) | `Q` | 於雙手手掌心匯聚奔騰的藍色雷電，並在 2 秒內快速動態放大 2.5 倍，即時跟隨手部移動。 |
| **水乱破の術** | 水遁 | 辰 (5) ➡️ 寅 (3) ➡️ 卯 (4) | `W` | 從手部噴射出奔流的水亂波特效，在 2 秒內動態放大 2.5 倍，維持 16:9 比例跟隨雙手。 |
| **龍火の術** | 火遁 | 巳 (6) ➡️ 辰 (5) ➡️ 卯 (4) ➡️ 寅 (3) | `E` | 手掌處引爆向前方猛烈噴發的龍火之術綠幕去背火焰，並跟隨手掌移動。 |
| **口寄せの術** | 通靈之術 | 戌 (11) ➡️ 亥 (12) ➡️ 酉 (10) ➡️ 申 (9) ➡️ 未 (8) | `T` | 結印完成先出現大面積濃煙霧，過 1 秒後通靈召喚獸從畫面中央淡入並膨脹至佔滿螢幕（以 Cover 比例維持長寬比，使用透明通道無損去背）。 |
| **口寄せ 土遁追牙の術** | 通靈/土遁 | 寅 (3) ➡️ 巳 (6) ➡️ 辰 (5) ➡️ 戌 (11) | `R` | 結印完成先出現大面積濃煙霧，過 1 秒後去背忍犬從畫面中央淡入並膨脹至佔滿整個螢幕（以 Cover 比例維持長寬比，使用 HSV 去背）。 |

---

## 📜 本專案所有內建忍術結印對照表 (標記 ⭐ 為配有特效的招式)

| 屬性/種類 | 忍術名稱 (日文) | 忍術名稱 (英文) | 結印順序 (由左至右依序比出) |
| :--- | :--- | :--- | :--- |
| **火遁** | 豪火球の術 | Fireball Jutsu | 巳 (6) ➡️ 未 (8) ➡️ 申 (9) ➡️ 亥 (12) ➡️ 午 (7) ➡️ 寅 (3) <br>*(或是：巳 (6) ➡️ 寅 (3) ➡️ 申 (9) ➡️ 亥 (12) ➡️ 午 (7) ➡️ 寅 (3))* |
| **火遁** | 鳳仙花の術 | Phoenix Flower Jutsu | 子 (1) ➡️ 寅 (3) ➡️ 戌 (11) ➡️ 丑 (2) ➡️ 卯 (4) ➡️ 寅 (3) |
| ⭐ **火遁** | **龍火の術** | Dragon Flame Jutsu | 巳 (6) ➡️ 辰 (5) ➡️ 卯 (4) ➡️ 寅 (3) |
| **火遁** | 火龍炎弾の術 | Dragon Flame Bomb | 未 (8) ➡️ 午 (7) ➡️ 巳 (6) ➡️ 辰 (5) ➡️ 子 (1) ➡️ 丑 (2) ➡️ 寅 (3) |
| ⭐ **水遁** | **水乱破の術** | Water Trumpet | 辰 (5) ➡️ 寅 (3) ➡️ 卯 (4) |
| **水遁** | 水鮫弾の術 | Water Shark Bomb Jutsu | 寅 (3) ➡️ 丑 (2) ➡️ 辰 (5) ➡️ 卯 (4) ➡️ 酉 (10) ➡️ 辰 (5) ➡️ 未 (8) |
| ⭐ **雷遁** | **千鳥** | Chidori | 丑 (2) ➡️ 卯 (4) ➡️ 申 (9) |
| ⭐ **基本忍術** | **分身の術** | Clone Jutsu | 未 (8) ➡️ 巳 (6) ➡️ 寅 (3) |                                                    
| **基本/通靈** | 変わり身の術 | Substitution Jutsu | 未 (8) ➡️ 亥 (12) ➡️ 丑 (2) ➡️ 戌 (11) ➡️ 巳 (6) |
| ⭐ **基本/通靈** | **口寄せの術** | Summoning Jutsu | 戌 (11) ➡️ 亥 (12) ➡️ 酉 (10) ➡️ 申 (9) ➡️ 未 (8) |
| ⭐ **通靈/土遁** | **口寄せ 土遁追牙の術** | Summoning: Earth Release: Tracking Fang Technique | 寅 (3) ➡️ 巳 (6) ➡️ 辰 (5) ➡️ 戌 (11) |
| **高級禁術** | 口寄せ 穢土転生の術 | Summoning: Impure World Reincarnation | 寅 (3) ➡️ 巳 (6) ➡️ 戌 (11) ➡️ 辰 (5) ➡️ 祈 (13) |
| **高級禁術** | 屍鬼封尽の術 | Sealing Jutsu: Reaper Death Seal | 巳 (6) ➡️ 亥 (12) ➡️ 未 (8) ➡️ 卯 (4) ➡️ 戌 (11) ➡️ 子 (1) ➡️ 酉 (10) ➡️ 午 (7) ➡️ 巳 (6) ➡️ 祈 (13) |

---

### 🌊 終極挑戰：水遁・水龍彈之術 (44 個印)

這是本專案中結印最複雜的術，您需要一氣呵成比完以下 **44 個手勢** 才能成功觸發：

> 丑 (2) ➡️ 申 (9) ➡️ 卯 (4) ➡️ 子 (1) ➡️ 亥 (12) ➡️ 酉 (10) ➡️ 丑 (2) ➡️ 午 (7) ➡️ 酉 (10) ➡️ 子 (1) ➡️ 寅 (3) ➡️ 戌 (11) ➡️ 寅 (3) ➡️ 巳 (6) ➡️ 丑 (2) ➡️ 未 (8) ➡️ 巳 (6) ➡️ 亥 (12) ➡️ 未 (8) ➡️ 子 (1) ➡️ 壬 (14) ➡️ 申 (9) ➡️ 酉 (10) ➡️ 辰 (5) ➡️ 酉 (10) ➡️ 丑 (2) ➡️ 午 (7) ➡️ 未 (8) ➡️ 寅 (3) ➡️ 巳 (6) ➡️ 子 (1) ➡️ 申 (9) ➡️ 卯 (4) ➡️ 亥 (12) ➡️ 辰 (5) ➡️ 未 (8) ➡️ 子 (1) ➡️ 丑 (2) ➡️ 申 (9) ➡️ 酉 (10) ➡️ 壬 (14) ➡️ 子 (1) ➡️ 亥 (12) ➡️ 酉 (10)

---

## 🖐️ 14 種手勢範例圖對照連結

如果您不清楚某個印的手勢，可以隨時點擊下方連結在瀏覽器中開啟範例圖片：

* **1. 子 (Rat)**：[點此看手勢圖](https://user-images.githubusercontent.com/37477845/95611897-6d032d00-0a9d-11eb-86c4-de1c50c0d7b6.jpg)
* **2. 丑 (Ox)**：[點此看手勢圖](https://user-images.githubusercontent.com/37477845/95611906-6ffe1d80-0a9d-11eb-9054-4e68c42e52ca.jpg)
* **3. 寅 (Tiger)**：[點此看手勢圖](https://user-images.githubusercontent.com/37477845/95611912-712f4a80-0a9d-11eb-8cb8-fc7097e16f60.jpg)
* **4. 卯 (Hare)**：[點此看手勢圖](https://user-images.githubusercontent.com/37477845/95611915-72607780-0a9d-11eb-9995-66524ce4f978.jpg)
* **5. 辰 (Dragon)**：[點此看手勢圖](https://user-images.githubusercontent.com/37477845/95611920-7391a480-0a9d-11eb-8e74-db39acf90f83.jpg)
* **6. 巳 (Snake)**：[點此看手勢圖](https://user-images.githubusercontent.com/37477845/95611922-742a3b00-0a9d-11eb-8a21-8bdf207db9bb.jpg)
* **7. 午 (Horse)**：[點此看手勢圖](https://user-images.githubusercontent.com/37477845/95611928-755b6800-0a9d-11eb-86c0-67605ffd6e9b.jpg)
* **8. 未 (Ram)**：[點此看手勢圖](https://user-images.githubusercontent.com/37477845/95611930-768c9500-0a9d-11eb-81c6-067b632dc43d.jpg)
* **9. 申 (Monkey)**：[點此看手勢圖](https://user-images.githubusercontent.com/37477845/95611931-77252b80-0a9d-11eb-97d6-e3efc6f1aac3.jpg)
* **10. 酉 (Bird)**：[點此看手勢圖](https://user-images.githubusercontent.com/37477845/95611935-77bdc200-0a9d-11eb-95e1-feb8bf7f61de.jpg)
* **11. 戌 (Dog)**：[點此看手勢圖](https://user-images.githubusercontent.com/37477845/95611936-78eeef00-0a9d-11eb-90b3-f565e4763c50.jpg)
* **12. 亥 (Boar)**：[點此看手勢圖](https://user-images.githubusercontent.com/37477845/95611938-7a201c00-0a9d-11eb-9d5f-1daf2405f20f.jpg)
* **壬 (Mizunoe)**：[點此看手勢圖](https://user-images.githubusercontent.com/37477845/95611947-7c827600-0a9d-11eb-97ae-9d7eabc58cd5.jpg)
* **祈 / 合掌 (Gassho)**：[點此看手勢圖](https://user-images.githubusercontent.com/37477845/95611943-7b514900-0a9d-11eb-97be-4fda80d17879.jpg)
