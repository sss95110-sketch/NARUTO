#!/usr/bin/env python
# -*- coding: utf-8 -*-

import csv
import time
import copy
import argparse
import os
from collections import deque

import cv2 as cv
import numpy as np

from model.yolox.yolox_onnx import YoloxONNX
from utils.cvfpscalc import CvFpsCalc

# 嘗試載入書法字體繪製工具，若失敗則使用內建 OpenCV 文字作為備份
try:
    from utils.cvdrawtext import CvDrawText
    HAS_CVDRAWTEXT = True
except ImportError:
    HAS_CVDRAWTEXT = False

FONT_PATH = './utils/font/衡山毛筆フォント.ttf'
if not os.path.exists(FONT_PATH):
    HAS_CVDRAWTEXT = False


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0, help="Camera device index")
    parser.add_argument("--width", help='Camera width', type=int, default=960)
    parser.add_argument("--height", help='Camera height', type=int, default=540)
    parser.add_argument("--model", type=str, default='model/yolox/yolox_nano.onnx', help='YOLOX model path')
    parser.add_argument('--score_th', type=float, default=0.7, help='Confidence threshold')
    parser.add_argument('--nms_th', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--nms_score_th', type=float, default=0.1, help='NMS Score threshold')
    parser.add_argument("--with_p6", action="store_true", help="Whether model uses p6 in FPN/PAN")
    return parser.parse_args()


class SmokeParticle:
    """濃郁煙霧粒子，模擬影分身召喚時的煙霧特效"""
    def __init__(self, x, y):
        self.x = x
        self.y = y
        # 隨機噴射角度與速度 (較慢且自然)
        angle = np.random.uniform(0, 2 * np.pi)
        speed = np.random.uniform(3.5, 8)
        self.vx = np.cos(angle) * speed
        self.vy = np.sin(angle) * speed - np.random.uniform(0.2, 0.8)  # 向上飄動
        self.radius = np.random.uniform(15, 65)   # 稍微增加起始半徑，以呈現厚重濃霧
        self.alpha = np.random.uniform(0.38, 0.68) # 顯著提高不透明度，讓煙霧更濃郁
        self.fade = np.random.uniform(0.06, 0.24) # 煙霧消散速度極快，約 0.1 秒消散
        self.growth = np.random.uniform(0.5, 1.1)   # 稍微增加膨脹速度
        # 煙霧主色調以豐富的淺灰白色為主
        c = np.random.randint(215, 245)
        self.color = (c, c, c)

    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.radius += self.growth
        self.alpha -= self.fade
        self.vx *= 0.94  # 空氣阻力
        self.vy *= 0.94
        return self.alpha > 0


def draw_smoke_particles(image, particles):
    """在影像上繪製半透明的煙霧粒子"""
    h, w, c = image.shape
    for p in particles:
        x, y, r = int(p.x), int(p.y), int(p.radius)
        if r <= 0 or p.alpha <= 0:
            continue
        
        # 計算邊界裁剪，防止陣列溢出
        x1 = max(0, x - r)
        y1 = max(0, y - r)
        x2 = min(w, x + r)
        y2 = min(h, y + r)
        if x2 <= x1 or y2 <= y1:
            continue
        
        # 建立圓形局部遮罩
        sub_w = x2 - x1
        sub_h = y2 - y1
        mask = np.zeros((sub_h, sub_w), dtype=np.uint8)
        cv.circle(mask, (x - x1, y - y1), r, 255, -1)
        
        # 進行 alpha 融合
        roi = image[y1:y2, x1:x2]
        color_bg = roi.copy()
        color_fg = np.full(roi.shape, p.color, dtype=np.uint8)
        
        blended = cv.addWeighted(color_fg, p.alpha, color_bg, 1.0 - p.alpha, 0)
        mask_bool = mask == 255
        image[y1:y2, x1:x2][mask_bool] = blended[mask_bool]


def blend_clone(live_frame, clone_roi, start_x, start_y, feather_pixels=45, alpha=1.0):
    """將分身區塊（包含背景與上下邊緣羽化）完美融合到實時畫面"""
    h, w, c = live_frame.shape
    clone_h, clone_w, clone_c = clone_roi.shape
    
    end_x = start_x + clone_w
    end_y = start_y + clone_h
    
    x1 = max(0, start_x)
    x2 = min(w, end_x)
    y1 = max(0, start_y)
    y2 = min(h, end_y)
    
    if x2 <= x1 or y2 <= y1:
        return
        
    roi_live = live_frame[y1:y2, x1:x2]
    roi_clone = clone_roi[(y1 - start_y):(y2 - start_y), (x1 - start_x):(x2 - start_x)]
    
    mask_h = y2 - y1
    mask_w = x2 - x1
    mask = np.ones((mask_h, mask_w), dtype=np.float32)
    
    # 左右邊界羽化 (面向中央人像的邊緣進行羽化)
    if start_x < w // 2:
        # 左分身：羽化右側邊界
        for i in range(min(feather_pixels, mask_w)):
            idx = mask_w - 1 - i
            mask[:, idx] = float(i) / feather_pixels
    else:
        # 右分身：羽化左側邊界
        for i in range(min(feather_pixels, mask_w)):
            mask[:, i] = float(i) / feather_pixels
            
    # 上下邊界羽化 (避免縮放後上下邊緣有生硬切口)
    vertical_feather = 15
    for i in range(min(vertical_feather, mask_h)):
        mask[i, :] *= float(i) / vertical_feather
        idx_b = mask_h - 1 - i
        mask[idx_b, :] *= float(i) / vertical_feather
            
    mask_3d = np.expand_dims(mask * alpha, axis=2)
    blended = roi_clone * mask_3d + roi_live * (1.0 - mask_3d)
    live_frame[y1:y2, x1:x2] = blended.astype(np.uint8)


def main():
    args = get_args()
    
    # 初始化攝影機與 ONNX 模型
    cap = cv.VideoCapture(args.device)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, args.height)
    
    yolox = YoloxONNX(
        model_path=args.model,
        input_shape=(416, 416),
        class_score_th=args.score_th,
        nms_th=args.nms_th,
        nms_score_th=args.nms_score_th,
        with_p6=args.with_p6,
    )
    
    # 讀取設定檔
    with open('setting/labels.csv', encoding='utf8') as f:
        labels = [row for row in csv.reader(f)]
        
    with open('setting/jutsu.csv', encoding='utf8') as f:
        jutsu = [row for row in csv.reader(f)]
        
    # 尋找「分身の術」的結印順序
    clone_jutsu_sequence = ""
    for row in jutsu:
        if len(row) > 2 and row[2] == "分身の術":
            clone_jutsu_sequence = "".join([s for s in row[4:] if s])
            break
    if not clone_jutsu_sequence:
        clone_jutsu_sequence = "未巳寅"  # 預設後備順序
        
    print(f"[*] 載入成功！「分身の術」觸發結印順序為：{' ➡️ '.join(clone_jutsu_sequence)}")
    print("[*] 提示：您也可以按下 [空白鍵] 或 [S 鍵] 直接手動觸發影分身特效。")
    print("[*] 提示：按下 [R 鍵] 可以重設並收回分身。")
    
    # 結印歷史紀錄
    sign_history_queue = deque(maxlen=44)
    sign_display_queue = deque(maxlen=18)
    sign_interval = 2.0  # 幾秒沒比印就清除紀錄
    sign_interval_start = 0.0
    
    # 消除手勢瞬間誤判抖動的 Queue
    chattering_check = 3
    chattering_check_queue = deque(maxlen=chattering_check)
    for i in range(chattering_check):
        chattering_check_queue.append(-1)
        
    fps_calc = CvFpsCalc()
    
    # 影分身核心狀態變數
    clones_active = False
    clone_start_time = 0.0
    smoke_particles = []
    
    window_name = "NARUTO Shadow Clone Jutsu Effect"
    cv.namedWindow(window_name, cv.WINDOW_NORMAL)
    
    while True:
        start_time = time.time()
        
        ret, frame = cap.read()
        if not ret:
            print("[!] 無法讀取攝影機畫面。")
            break
            
        frame = cv.flip(frame, 1)
        h, w, c = frame.shape
        debug_image = frame.copy()
        
        # 1. 執行手勢偵測
        bboxes, scores, class_ids = yolox.inference(frame)
        
        # 找出當前信心度最高的手勢
        best_class_id = -1
        best_score = 0.0
        for bbox, score, class_id in zip(bboxes, scores, class_ids):
            if score >= args.score_th and score > best_score:
                best_score = score
                best_class_id = int(class_id) + 1  # 配合 labels.csv 對應 (1-indexed)
                
        # 2. 處理結印手勢消除抖動與加入 Queue
        if best_class_id != -1:
            chattering_check_queue.append(best_class_id)
            if len(set(chattering_check_queue)) == 1:
                registered_id = chattering_check_queue[0]
                if len(sign_display_queue) == 0 or sign_display_queue[-1] != registered_id:
                    sign_display_queue.append(registered_id)
                    sign_history_queue.append(registered_id)
                    sign_interval_start = time.time()
        else:
            chattering_check_queue.append(-1)
            
        # 超時清除手勢紀錄
        if (time.time() - sign_interval_start) > sign_interval:
            sign_display_queue.clear()
            sign_history_queue.clear()
            
        # 3. 檢查結印歷史是否觸發了「分身の術」
        triggered = False
        sign_history_str = ""
        if len(sign_history_queue) > 0:
            sign_history_str = "".join([labels[sid][1] for sid in sign_history_queue])
            if sign_history_str.endswith(clone_jutsu_sequence):
                triggered = True
                sign_history_queue.clear()
                sign_display_queue.clear()
                
        # 4. 監聽鍵盤事件 (Space 或 S 觸發，R 重設，ESC 或 Q 退出)
        key = cv.waitKey(1) & 0xFF
        if key == ord(' ') or key == ord('s') or key == ord('S'):
            triggered = True
        elif key == ord('r') or key == ord('R'):
            clones_active = False
            smoke_particles.clear()
            print("[*] 影分身解除。")
        elif key == 27 or key == ord('q') or key == ord('Q'):
            break
            
        # 5. 觸發影分身效果
        if triggered:
            print("[*] 影分身之術 ！！！")
            clones_active = True
            clone_start_time = time.time()
            
            # 依據分身的高度尺寸與 y 偏移量來產生細緻煙霧粒子，讓煙霧落在正確的區域
            clone_scale = 0.82
            clone_h = int(h * clone_scale)
            clone_w = int((w // 3) * clone_scale)
            start_y = int((h - clone_h) * 0.4)
            
            smoke_particles.clear()
            # 單邊粒子數量增加至 200 個以提高整體煙霧濃度
            for _ in range(200):
                # 左分身煙霧
                py = np.random.uniform(start_y, start_y + clone_h)
                px = np.random.uniform((w // 3 - clone_w) // 2, (w // 3 - clone_w) // 2 + clone_w)
                smoke_particles.append(SmokeParticle(px, py))
            for _ in range(200):
                # 右分身煙霧
                py = np.random.uniform(start_y, start_y + clone_h)
                px = np.random.uniform(2 * w // 3 + (w // 3 - clone_w) // 2, 2 * w // 3 + (w // 3 - clone_w) // 2 + clone_w)
                smoke_particles.append(SmokeParticle(px, py))
                
        # 6. 渲染影分身特效與實時畫面結合
        output_frame = debug_image.copy()
        
        if clones_active:
            elapsed_time = time.time() - clone_start_time
            
            # 擷取人像所在的中央 1/3 區塊 (全高)
            crop_x1 = w // 3
            crop_x2 = 2 * w // 3
            
            # 實時同步：直接複製當前實時 live 影像的中央區塊
            raw_left = frame[0:h, crop_x1:crop_x2]
            raw_right = frame[0:h, crop_x1:crop_x2]
            
            # 縮放分身的大小 (設為原高的 82%，原寬的 82%)，產生高度/大小差異以增添深度感
            clone_scale = 0.82
            clone_h = int(h * clone_scale)
            clone_w = int((w // 3) * clone_scale)
            
            left_roi = cv.resize(raw_left, (clone_w, clone_h), interpolation=cv.INTER_LINEAR)
            right_roi = cv.resize(raw_right, (clone_w, clone_h), interpolation=cv.INTER_LINEAR)
            
            # 設定分身的繪製位置 (左/右置中，且垂直位置略微向上偏移營造後退的空間感)
            start_y = int((h - clone_h) * 0.4)
            left_x = (w // 3 - clone_w) // 2
            right_x = 2 * w // 3 + (w // 3 - clone_w) // 2
            
            # 前 0.5 秒漸變出現
            alpha = min(1.0, elapsed_time / 0.5)
            
            # 將分身融合至實時畫面的左側與右側
            blend_clone(output_frame, left_roi, left_x, start_y, feather_pixels=45, alpha=alpha)
            blend_clone(output_frame, right_roi, right_x, start_y, feather_pixels=45, alpha=alpha)

                    
        # 7. 更新並繪製煙霧粒子
        if smoke_particles:
            # 更新粒子狀態，保留生命值大於 0 的粒子
            smoke_particles = [p for p in smoke_particles if p.update()]
            draw_smoke_particles(output_frame, smoke_particles)
            
        # 8. 繪製手勢辨識框與偵測歷史 (印的排版)
        for bbox, score, class_id in zip(bboxes, scores, class_ids):
            class_id = int(class_id) + 1
            if score < args.score_th:
                continue
            x1, y1 = int(bbox[0]), int(bbox[1])
            x2, y2 = int(bbox[2]), int(bbox[3])
            cv.rectangle(output_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv.putText(
                output_frame, f"{labels[class_id][1]} ({score:.2f})", 
                (x1, y1 - 10), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv.LINE_AA
            )
            
        # 9. 顯示當前結印進度與資訊於畫面下方
        fps_val = fps_calc.get()
        cv.putText(
            output_frame, f"FPS: {fps_val}", (10, 30), 
            cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv.LINE_AA
        )
        
        sign_display = " ".join([labels[sid][1] for sid in sign_display_queue])
        cv.putText(
            output_frame, f"Signs: {sign_display}", (10, h - 20), 
            cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv.LINE_AA
        )
        
        # 10. 置中與等比例縮放視窗
        try:
            rect = cv.getWindowImageRect(window_name)
        except Exception:
            rect = None
            
        if rect is not None and rect[2] > 0 and rect[3] > 0:
            win_w, win_h = rect[2], rect[3]
            src_h, src_w = output_frame.shape[:2]
            aspect_ratio = src_w / src_h
            win_aspect = win_w / win_h
            
            if win_aspect > aspect_ratio:
                new_h = win_h
                new_w = int(win_h * aspect_ratio)
            else:
                new_w = win_w
                new_h = int(win_w / aspect_ratio)
                
            new_w = max(1, new_w)
            new_h = max(1, new_h)
            resized_image = cv.resize(output_frame, (new_w, new_h), interpolation=cv.INTER_AREA)
            
            canvas = np.zeros((win_h, win_w, 3), dtype=np.uint8)
            x_offset = (win_w - new_w) // 2
            y_offset = (win_h - new_h) // 2
            canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized_image
            cv.imshow(window_name, canvas)
        else:
            cv.imshow(window_name, output_frame)
            
        # 控制幀率
        elapsed_time = time.time() - start_time
        sleep_time = max(0.001, (1.0 / 30.0) - elapsed_time)
        time.sleep(sleep_time)
        
    cap.release()
    cv.destroyAllWindows()


if __name__ == '__main__':
    main()