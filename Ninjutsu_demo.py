#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import csv
import time
import copy
from collections import deque

import cv2 as cv
import numpy as np

from utils import CvFpsCalc
from utils import CvDrawText
from model.yolox.yolox_onnx import YoloxONNX
from PIL import Image, ImageSequence

state = 'COVER'
button_area = (0, 0, 0, 0)


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--width", help='cap width', type=int, default=960)
    parser.add_argument("--height", help='cap height', type=int, default=540)
    parser.add_argument("--file", type=str, default=None)

    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--skip_frame", type=int, default=0)

    parser.add_argument(
        "--model",
        type=str,
        default='model/yolox/yolox_nano.onnx',
    )
    parser.add_argument(
        '--input_shape',
        type=str,
        default="416,416",
        help="Specify an input shape for inference.",
    )
    parser.add_argument( #判定門檻 預設是  0.8 （AI 只要有 80% 信心就成立）#
        '--score_th',
        type=float,
        default=0.8,
        help='Class confidence',
    )
    parser.add_argument(
        '--nms_th',
        type=float,
        default=0.45,
        help='NMS IoU threshold',
    )
    parser.add_argument(
        '--nms_score_th',
        type=float,
        default=0.1,
        help='NMS Score threshold',
    )
    parser.add_argument(
        "--with_p6",
        action="store_true",
        help="Whether your model uses p6 in FPN/PAN.",
    )

    parser.add_argument("--sign_interval", type=float, default=2.0)
    parser.add_argument("--jutsu_display_time", type=int, default=5)

    parser.add_argument("--use_display_score", type=bool, default=False)
    parser.add_argument("--erase_bbox", type=bool, default=False)
    parser.add_argument("--use_jutsu_lang_en", type=bool, default=False)

    parser.add_argument("--chattering_check", type=int, default=1)

    parser.add_argument("--use_fullscreen", type=bool, default=False)

    args = parser.parse_args()

    return args


class GIFReader:
    """用 Pillow 載入 GIF，並為 OpenCV 提供帶有 Alpha 通道去背的影格"""
    def __init__(self, filepath):
        self.frames = []
        try:
            im = Image.open(filepath)
            for frame in ImageSequence.Iterator(im):
                frame_rgba = frame.convert('RGBA')
                arr = np.array(frame_rgba)
                # 分離出 BGR 與 Alpha 通道
                bgr = cv.cvtColor(arr[:, :, :3], cv.COLOR_RGB2BGR)
                alpha = arr[:, :, 3]
                self.frames.append((bgr, alpha))
        except Exception as e:
            print(f"[!] 載入 GIF 失敗: {filepath}, 錯誤: {e}")
        self.idx = 0

    def read(self):
        if not self.frames:
            return False, None, None
        bgr, alpha = self.frames[self.idx]
        self.idx = (self.idx + 1) % len(self.frames)
        return True, bgr, alpha


class SmokeParticle:
    """濃郁煙霧粒子，模擬影分身召喚時的煙霧特效"""
    def __init__(self, x, y):
        self.x = x
        self.y = y
        # 隨機噴射角度與速度
        angle = np.random.uniform(0, 2 * np.pi)
        speed = np.random.uniform(3.5, 8)
        self.vx = np.cos(angle) * speed
        self.vy = np.sin(angle) * speed - np.random.uniform(0.2, 0.8)  # 向上飄動
        self.radius = np.random.uniform(15, 65)   # 稍微增加起始半徑，以呈現厚重濃霧
        self.alpha = np.random.uniform(0.38, 0.68) # 顯著提高不透明度，讓煙霧更濃郁
        self.fade = np.random.uniform(0.008, 0.016) # 縮短消散時間 0.3 秒
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


def draw_smoke_particles(image, smoke_particles):
    """在影像上繪製半透明的煙霧粒子"""
    h, w, c = image.shape
    for p in smoke_particles:
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
    # 引数解析 #################################################################
    args = get_args()

    cap_width = args.width
    cap_height = args.height
    cap_device = args.device
    if args.file is not None:  # 動画ファイルを利用する場合
        cap_device = args.file

    fps = args.fps
    skip_frame = args.skip_frame

    model_path = args.model
    input_shape = tuple(map(int, args.input_shape.split(',')))
    score_th = args.score_th
    nms_th = args.nms_th
    nms_score_th = args.nms_score_th
    with_p6 = args.with_p6

    sign_interval = args.sign_interval
    jutsu_display_time = args.jutsu_display_time

    use_display_score = args.use_display_score
    erase_bbox = args.erase_bbox
    use_jutsu_lang_en = args.use_jutsu_lang_en

    chattering_check = args.chattering_check

    use_fullscreen = args.use_fullscreen

    # カメラ準備 (延遲到按鈕點擊後初始化) ######################################
    cap = None

    # モデル読み込み ############################################################
    yolox = YoloxONNX(
        model_path=model_path,
        input_shape=input_shape,
        class_score_th=score_th,
        nms_th=nms_th,
        nms_score_th=nms_score_th,
        with_p6=with_p6,
        # providers=['CPUExecutionProvider'],
    )

    # FPS計測モジュール #########################################################
    cvFpsCalc = CvFpsCalc()

    # フォント読み込み ##########################################################
    # https://opentype.jp/kouzanmouhitufont.htm
    font_path = './utils/font/衡山毛筆フォント.ttf'

    # ラベル読み込み ###########################################################
    with open('setting/labels.csv', encoding='utf8') as f:  # 印
        labels = csv.reader(f)
        labels = [row for row in labels]

    with open('setting/jutsu.csv', encoding='utf8') as f:  # 術
        jutsu = csv.reader(f)
        jutsu = [row for row in jutsu]

    # 印の表示履歴および、検出履歴 ##############################################
    sign_max_display = 18
    sign_max_history = 44
    sign_display_queue = deque(maxlen=sign_max_display)
    sign_history_queue = deque(maxlen=sign_max_history)

    chattering_check_queue = deque(maxlen=chattering_check)
    for index in range(-1, -1 - chattering_check, -1):
        chattering_check_queue.append(index)

    # 術名の言語設定 ###########################################################
    lang_offset = 0
    jutsu_font_size_ratio = sign_max_display
    if use_jutsu_lang_en:
        lang_offset = 1
        jutsu_font_size_ratio = int((sign_max_display / 3) * 4)

    # その他変数初期化 #########################################################
    sign_interval_start = 0  # 印のインターバル開始時間初期化
    jutsu_index = 0  # 術表示名のインデックス
    jutsu_start_time = 0  # 術名表示の開始時間初期化
    frame_count = 0  # フレームナンバーカウンタ

    # 特效播放變數
    is_playing_effect = False
    effect_cap = None
    effect_start_time = 0.0
    
    # 龍火之術變數
    is_playing_dragon = False
    dragon_cap = None
    dragon_start_time = 0.0
    
    # 土遁追牙之術變數
    is_playing_dog = False
    dog_image = None
    dog_start_time = 0.0
    
    # 通靈之術變數
    is_playing_summon = False
    summon_image = None
    summon_start_time = 0.0
    
    last_jutsu_index = 0
    last_hand_cx = 480
    last_hand_cy = 270

    # 影分身特效變數
    clones_active = False
    clone_start_time = 0.0
    smoke_particles = []

    # 千鳥特效變數
    is_playing_chidori = False
    chidori_start_time = 0.0
    chidori_reader = None

    window_name = 'NARUTO HandSignDetection Ninjutsu Demo'
    cv.namedWindow(window_name, cv.WINDOW_NORMAL)

    # 封面載入與按鈕繪製 #######################################################
    cover_image_raw = cv.imread('VisualEffects/Cover.png')
    if cover_image_raw is None:
        cover_image_raw = np.zeros((540, 960, 3), dtype=np.uint8)
        cover_image_raw = CvDrawText.puttext(cover_image_raw, "Cover.png Not Found", (280, 250), font_path, 30, (255, 255, 255))
    else:
        cover_image_raw = cv.resize(cover_image_raw, (960, 540))

    # 在 cover_image_raw 上繪製半透明紅色開始按鈕與白色邊框 (往下移至 y: 330~410)
    overlay = cover_image_raw.copy()
    cv.rectangle(overlay, (380, 330), (580, 410), (0, 0, 200), -1)
    cv.rectangle(overlay, (380, 330), (580, 410), (255, 255, 255), 2)
    cover_image_raw = cv.addWeighted(overlay, 0.7, cover_image_raw, 0.3, 0)
    
    # 寫上毛筆字 "START" (y 坐標改為 345，置中位置)
    cover_image_raw = CvDrawText.puttext(cover_image_raw, "START", (430, 345), font_path, 32, (255, 255, 255))

    # 滑鼠點擊按鈕事件 callback
    def on_mouse(event, x, y, flags, param):
        global state, button_area
        if state == 'COVER' and event == cv.EVENT_LBUTTONDOWN:
            bx1, by1, bx2, by2 = button_area
            if bx1 <= x <= bx2 and by1 <= y <= by2:
                state = 'INIT_CAMERA'

    cv.setMouseCallback(window_name, on_mouse)

    while True:
        global state, button_area

        if state == 'COVER':
            try:
                rect = cv.getWindowImageRect(window_name)
            except Exception:
                rect = None

            if rect is not None and rect[2] > 0 and rect[3] > 0:
                win_w, win_h = rect[2], rect[3]
                src_h, src_w = cover_image_raw.shape[:2]
                
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
                
                resized_image = cv.resize(cover_image_raw, (new_w, new_h), interpolation=cv.INTER_AREA)
                
                canvas = np.zeros((win_h, win_w, 3), dtype=np.uint8)
                x_offset = (win_w - new_w) // 2
                y_offset = (win_h - new_h) // 2
                canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized_image
                button_area = (
                    x_offset + int(new_w * (380 / 960)),
                    y_offset + int(new_h * (330 / 540)),
                    x_offset + int(new_w * (580 / 960)),
                    y_offset + int(new_h * (410 / 540))
                )
                
                cv.imshow(window_name, canvas)
            else:
                cv.imshow(window_name, cover_image_raw)
                button_area = (380, 330, 580, 410)

            key = cv.waitKey(50)
            if key == 27:  # ESC 離開
                break
            continue

        elif state == 'INIT_CAMERA':
            # 點擊按鈕後，在此處才真正初始化相機
            cap = cv.VideoCapture(cap_device)
            cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
            cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)
            state = 'PLAY'
            continue

        start_time = time.time()

        # カメラキャプチャ #####################################################
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv.flip(frame, 1)
        frame_count += 1
        debug_image = copy.deepcopy(frame)



        # FPS計測 ##############################################################
        fps_result = cvFpsCalc.get()

        # 検出実施 #############################################################
        bboxes, scores, class_ids = yolox.inference(frame)

        # 更新當前手勢中心點坐標 (供下一影格的特效跟隨使用)
        current_hands = []
        best_bbox = None
        best_score = 0
        # 播放千鳥、水遁、龍火、土遁追牙或通靈之術時，降低手部追隨的置信度門檻至 0.15，以確保單手揮動置信度極低時依然能靈敏跟隨
        current_follow_th = 0.15 if (is_playing_chidori or is_playing_effect or is_playing_dragon or is_playing_dog or is_playing_summon) else score_th
        for bbox, score, class_id in zip(bboxes, scores, class_ids):
            if score >= current_follow_th:
                cx = int((bbox[0] + bbox[2]) / 2)
                cy = int((bbox[1] + bbox[3]) / 2)
                current_hands.append((cx, cy))
                if score > best_score:
                    best_score = score
                    best_bbox = bbox
        if best_bbox is not None:
            last_hand_cx = int((best_bbox[0] + best_bbox[2]) / 2)
            last_hand_cy = int((best_bbox[1] + best_bbox[3]) / 2)

        # 検出内容の履歴追加 ####################################################
        for _, score, class_id in zip(bboxes, scores, class_ids):
            class_id = int(class_id) + 1

            # 検出閾値未満の結果は捨てる
            if score < score_th:
                continue

            # 指定回数以上、同じ印が続いた場合に、印検出とみなす ※瞬間的な誤検出対策
            chattering_check_queue.append(class_id)
            if len(set(chattering_check_queue)) != 1:
                continue

            # 前回と異なる印の場合のみキューに登録
            if len(sign_display_queue
                   ) == 0 or sign_display_queue[-1] != class_id:
                sign_display_queue.append(class_id)
                sign_history_queue.append(class_id)
                sign_interval_start = time.time()  # 印の最終検出時間

        # 前回の印検出から指定時間が経過した場合、履歴を消去 ####################
        if (time.time() - sign_interval_start) > sign_interval:
            sign_display_queue.clear()
            sign_history_queue.clear()

        # 術成立判定 #########################################################
        jutsu_index, jutsu_start_time, matched = check_jutsu(
            sign_history_queue,
            labels,
            jutsu,
            jutsu_index,
            jutsu_start_time,
        )

        # 檢查是否觸發特定忍術特效 (包含分身之術與水亂波之術)
        if matched:
            if jutsu[jutsu_index][2] == "分身の術":
                clones_active = True
                clone_start_time = time.time()
                
                h, w = frame.shape[:2]
                clone_scale = 0.82
                clone_h = int(h * clone_scale)
                clone_w = int((w // 3) * clone_scale)
                start_y = int((h - clone_h) * 0.4)
                
                smoke_particles.clear()
                # 產生細緻煙霧粒子 (單邊減少20%粒子，改為 56 個粒子以防發盪)
                for _ in range(56):
                    py = np.random.uniform(start_y, start_y + clone_h)
                    px = np.random.uniform((w // 3 - clone_w) // 2, (w // 3 - clone_w) // 2 + clone_w)
                    smoke_particles.append(SmokeParticle(px, py))
                for _ in range(56):
                    py = np.random.uniform(start_y, start_y + clone_h)
                    px = np.random.uniform(2 * w // 3 + (w // 3 - clone_w) // 2, 2 * w // 3 + (w // 3 - clone_w) // 2 + clone_w)
                    smoke_particles.append(SmokeParticle(px, py))
                    
            elif jutsu[jutsu_index][2] == "水乱破の術":
                if effect_cap is not None:
                    effect_cap.release()
                effect_cap = cv.VideoCapture('VisualEffects/LQ.mov')
                is_playing_effect = True
                effect_start_time = time.time()
            elif jutsu[jutsu_index][2] == "千鳥":
                chidori_reader = GIFReader('VisualEffects/lightning-aura.gif')
                is_playing_chidori = True
                chidori_start_time = time.time()
            elif jutsu[jutsu_index][2] == "龍火の術":
                if dragon_cap is not None:
                    dragon_cap.release()
                dragon_cap = cv.VideoCapture('VisualEffects/boom.mp4')
                is_playing_dragon = True
                dragon_start_time = time.time()
            elif jutsu[jutsu_index][2] == "口寄せ 土遁追牙の術":
                dog_image = cv.imread('VisualEffects/dogdogdog.png')
                is_playing_dog = True
                dog_start_time = time.time()
                h, w = frame.shape[:2]
                smoke_particles.clear()
                # 增加粒子數量以呈現大面積煙霧，並以畫面中央為中心
                for _ in range(120):
                    py = np.random.uniform(h // 2 - 120, h // 2 + 120)
                    px = np.random.uniform(w // 2 - 180, w // 2 + 180)
                    smoke_particles.append(SmokeParticle(px, py))
            elif jutsu[jutsu_index][2] == "口寄せの術":
                summon_image = cv.imread('VisualEffects/MPKa7iiQyaoGxbGagC6Fea (1).png', cv.IMREAD_UNCHANGED)
                is_playing_summon = True
                summon_start_time = time.time()
                h, w = frame.shape[:2]
                smoke_particles.clear()
                # 增加粒子數量以呈現大面積煙霧，並以畫面中央為中心
                for _ in range(120):
                    py = np.random.uniform(h // 2 - 120, h // 2 + 120)
                    px = np.random.uniform(w // 2 - 180, w // 2 + 180)
                    smoke_particles.append(SmokeParticle(px, py))
            # 觸發術後，清空印的顯示佇列
            sign_display_queue.clear()

        # キー処理 ###########################################################
        key = cv.waitKey(1)
        if key == 99:  # C：印の履歴を消去
            sign_display_queue.clear()
            sign_history_queue.clear()
            
        # 手動測試按鍵 (方便測試特效，後續可拿掉)
        if key == 32 or key == 115:  # 空白鍵(32) 或 S鍵(115)：手動觸發影分身之術
            clones_active = True
            clone_start_time = time.time()
            h, w = frame.shape[:2]
            clone_scale = 0.82
            clone_h = int(h * clone_scale)
            clone_w = int((w // 3) * clone_scale)
            start_y = int((h - clone_h) * 0.4)
            smoke_particles.clear()
            for _ in range(56):
                py = np.random.uniform(start_y, start_y + clone_h)
                px = np.random.uniform((w // 3 - clone_w) // 2, (w // 3 - clone_w) // 2 + clone_w)
                smoke_particles.append(SmokeParticle(px, py))
            for _ in range(56):
                py = np.random.uniform(start_y, start_y + clone_h)
                px = np.random.uniform(2 * w // 3 + (w // 3 - clone_w) // 2, 2 * w // 3 + (w // 3 - clone_w) // 2 + clone_w)
                smoke_particles.append(SmokeParticle(px, py))
            # 顯示招式名稱
            for idx, item in enumerate(jutsu):
                if len(item) > 2 and item[2] == "分身の術":
                    jutsu_index = idx
                    jutsu_start_time = time.time()
                    break

        elif key == 113:  # Q鍵(113)：手動觸發千鳥特效
            chidori_reader = GIFReader('VisualEffects/lightning-aura.gif')
            is_playing_chidori = True
            chidori_start_time = time.time()
            # 顯示招式名稱
            for idx, item in enumerate(jutsu):
                if len(item) > 2 and item[2] == "千鳥":
                    jutsu_index = idx
                    jutsu_start_time = time.time()
                    break

        elif key == 119:  # W鍵(119)：手動觸發水遁水乱破之術特效
            if effect_cap is not None:
                effect_cap.release()
            effect_cap = cv.VideoCapture('VisualEffects/LQ.mov')
            is_playing_effect = True
            effect_start_time = time.time()
            # 顯示招式名稱
            for idx, item in enumerate(jutsu):
                if len(item) > 2 and item[2] == "水乱破の術":
                    jutsu_index = idx
                    jutsu_start_time = time.time()
                    break

        elif key == 101:  # E鍵(101)：手動觸發火遁龍火之術特效
            if dragon_cap is not None:
                dragon_cap.release()
            dragon_cap = cv.VideoCapture('VisualEffects/boom.mp4')
            is_playing_dragon = True
            dragon_start_time = time.time()
            # 顯示招式名稱
            for idx, item in enumerate(jutsu):
                if len(item) > 2 and item[2] == "龍火の術":
                    jutsu_index = idx
                    jutsu_start_time = time.time()
                    break

        elif key == 114:  # R鍵(114)：手動觸發口寄せ 土遁追牙の術特效
            dog_image = cv.imread('VisualEffects/dogdogdog.png')
            is_playing_dog = True
            dog_start_time = time.time()
            h, w = frame.shape[:2]
            smoke_particles.clear()
            # 增加粒子數量以呈現大面積煙霧，並以畫面中央為中心
            for _ in range(120):
                py = np.random.uniform(h // 2 - 120, h // 2 + 120)
                px = np.random.uniform(w // 2 - 180, w // 2 + 180)
                smoke_particles.append(SmokeParticle(px, py))
            # 顯示招式名稱
            for idx, item in enumerate(jutsu):
                if len(item) > 2 and item[2] == "口寄せ 土遁追牙の術":
                    jutsu_index = idx
                    jutsu_start_time = time.time()
                    break
        elif key == 116:  # T鍵(116)：手動觸發口寄せの術特效
            summon_image = cv.imread('VisualEffects/MPKa7iiQyaoGxbGagC6Fea (1).png', cv.IMREAD_UNCHANGED)
            is_playing_summon = True
            summon_start_time = time.time()
            h, w = frame.shape[:2]
            smoke_particles.clear()
            # 增加粒子數量以呈現大面積煙霧，並以畫面中央為中心
            for _ in range(120):
                py = np.random.uniform(h // 2 - 120, h // 2 + 120)
                px = np.random.uniform(w // 2 - 180, w // 2 + 180)
                smoke_particles.append(SmokeParticle(px, py))
            # 顯示招式名稱
            for idx, item in enumerate(jutsu):
                if len(item) > 2 and item[2] == "口寄せの術":
                    jutsu_index = idx
                    jutsu_start_time = time.time()
                    break

        if key == 27:  # ESC：プログラム終了
            break

        # 画面反映 #############################################################
        
        # 渲染影分身特效與實時畫面結合 (影分身持續 6 秒)
        if clones_active:
            elapsed_time = time.time() - clone_start_time
            if elapsed_time < 6.0:
                h, w = frame.shape[:2]
                crop_x1 = w // 3
                crop_x2 = 2 * w // 3
                raw_left = frame[0:h, crop_x1:crop_x2]
                raw_right = frame[0:h, crop_x1:crop_x2]
                
                clone_scale = 0.82
                clone_h = int(h * clone_scale)
                clone_w = int((w // 3) * clone_scale)
                
                left_roi = cv.resize(raw_left, (clone_w, clone_h), interpolation=cv.INTER_LINEAR)
                right_roi = cv.resize(raw_right, (clone_w, clone_h), interpolation=cv.INTER_LINEAR)
                
                start_y = int((h - clone_h) * 0.4)
                left_x = (w // 3 - clone_w) // 2
                right_x = 2 * w // 3 + (w // 3 - clone_w) // 2
                
                alpha = min(1.0, elapsed_time / 0.5)
                
                blend_clone(debug_image, left_roi, left_x, start_y, feather_pixels=45, alpha=alpha)
                blend_clone(debug_image, right_roi, right_x, start_y, feather_pixels=45, alpha=alpha)
            else:
                clones_active = False
                
        # 更新並繪製煙霧粒子
        if smoke_particles:
            smoke_particles = [p for p in smoke_particles if p.update()]
            draw_smoke_particles(debug_image, smoke_particles)

        # 渲染水遁特效 (LQ.mov) - 跟隨所有手部位置並縮小且維持原始 16:9 長寬比
        if is_playing_effect and effect_cap is not None:
            ret_e, effect_frame = effect_cap.read()
            if not ret_e:
                effect_cap.release()
                effect_cap = None
                is_playing_effect = False
            else:
                h, w = debug_image.shape[:2]
                
                # 計算播放經過的時間，並在 2 秒內動態放大至 250%
                dt_e = time.time() - effect_start_time
                scale_ratio = min(1.0, dt_e / 2.0)
                scale = 1.0 + 1.5 * scale_ratio
                
                # 根據影片原始寬高比動態計算尺寸，防止畫面被壓扁成橢圓
                eff_h, eff_w = effect_frame.shape[:2]
                aspect_ratio = eff_w / eff_h
                eh = int(250 * scale)
                ew = int(eh * aspect_ratio)
                
                effect_resized = cv.resize(effect_frame, (ew, eh))
                
                # 決定要貼上特效的所有中心點
                # 如果當前有偵測到合格的手部，則在所有手部中心貼上特效；若無，則使用最後記錄的位置
                targets = current_hands if len(current_hands) > 0 else [(last_hand_cx, last_hand_cy)]
                
                for cx, cy in targets:
                    x1 = max(0, cx - ew // 2)
                    y1 = max(0, cy - eh // 2)
                    x2 = min(w, cx + ew // 2)
                    y2 = min(h, cy + eh // 2)
                    
                    # 裁剪以防溢出邊界
                    crop_x1 = ew // 2 - (cx - x1)
                    crop_y1 = eh // 2 - (cy - y1)
                    crop_x2 = ew // 2 + (x2 - cx)
                    crop_y2 = eh // 2 + (y2 - cy)
                    
                    if (x2 > x1) and (y2 > y1):
                        roi = debug_image[y1:y2, x1:x2]
                        eff_cropped = effect_resized[crop_y1:crop_y2, crop_x1:crop_x2]
                        # 水遁去背使用 cv.add (因 LQ.mov 背景為純黑)
                        debug_image[y1:y2, x1:x2] = cv.add(roi, eff_cropped)

        # 渲染火遁龍火之術 (boom.mp4) - 跟隨所有手部位置並在 2 秒內縮放至 400% (綠幕 Chroma Keying 去背)
        if is_playing_dragon and dragon_cap is not None:
            ret_d, dragon_frame = dragon_cap.read()
            if not ret_d:
                dragon_cap.release()
                dragon_cap = None
                is_playing_dragon = False
            else:
                h, w = debug_image.shape[:2]
                
                # 計算播放經過的時間，並在 2 秒內動態放大至 400%
                dt_d = time.time() - dragon_start_time
                scale_ratio = min(1.0, dt_d / 2.0)
                scale = 1.0 + 3.0 * scale_ratio
                
                # 根據影片原始寬高比動態計算尺寸
                eff_h, eff_w = dragon_frame.shape[:2]
                aspect_ratio = eff_w / eff_h
                eh = int(250 * scale)
                ew = int(eh * aspect_ratio)
                
                dragon_resized = cv.resize(dragon_frame, (ew, eh))
                
                # 對調整大小後的影格進行綠幕去背 (Chroma Keying)
                # 轉換至 HSV 空間
                hsv = cv.cvtColor(dragon_resized, cv.COLOR_BGR2HSV)
                # 綠幕 HSV 範圍設定
                lower_green = np.array([35, 40, 40])
                upper_green = np.array([85, 255, 255])
                green_mask = cv.inRange(hsv, lower_green, upper_green)
                # 火焰主體遮罩 (反轉綠色遮罩)
                fg_mask = cv.bitwise_not(green_mask)
                
                # 決定要貼上特效的所有中心點
                targets = current_hands if len(current_hands) > 0 else [(last_hand_cx, last_hand_cy)]
                
                for cx, cy in targets:
                    x1 = max(0, cx - ew // 2)
                    y1 = max(0, cy - eh // 2)
                    x2 = min(w, cx + ew // 2)
                    y2 = min(h, cy + eh // 2)
                    
                    # 裁剪以防溢出邊界
                    crop_x1 = ew // 2 - (cx - x1)
                    crop_y1 = eh // 2 - (cy - y1)
                    crop_x2 = ew // 2 + (x2 - cx)
                    crop_y2 = eh // 2 + (y2 - cy)
                    
                    if (x2 > x1) and (y2 > y1):
                        roi = debug_image[y1:y2, x1:x2]
                        eff_cropped = dragon_resized[crop_y1:crop_y2, crop_x1:crop_x2]
                        mask_cropped = fg_mask[crop_y1:crop_y2, crop_x1:crop_x2]
                        
                        # 使用綠幕去背遮罩進行 alpha 混合
                        mask_f = mask_cropped.astype(np.float32) / 255.0
                        mask_f = np.expand_dims(mask_f, axis=2)
                        
                        # 融合公式
                        blended = eff_cropped.astype(np.float32) * mask_f + roi.astype(np.float32) * (1.0 - mask_f)
                        debug_image[y1:y2, x1:x2] = blended.astype(np.uint8)

        # 渲染口寄せ 土遁追牙之術 (dogdogdog.png) - 固定於畫面中央並在 2 秒內縮放至佔滿整個視窗 (黑白棋盤格去背與淡入混合，前 1 秒僅顯示煙霧)
        if is_playing_dog and dog_image is not None:
            dt_dog = time.time() - dog_start_time
            if dt_dog < 6.0:
                h, w = debug_image.shape[:2]
                
                # 前 1 秒僅顯示煙霧，1 秒後才開始淡入並放大忍犬
                if dt_dog >= 1.0:
                    t_rel = dt_dog - 1.0
                    # 2 秒內從初始尺寸動態膨脹至佔滿整個視窗
                    scale_ratio = min(1.0, t_rel / 2.0)
                    
                    eff_h, eff_w = dog_image.shape[:2]
                    aspect_ratio = eff_w / eff_h
                    
                    # 計算 Cover 縮放：保持 1.833 原始長寬比且長寬皆大於等於視窗大小
                    scale_h = h / eff_h
                    scale_w = w / eff_w
                    cover_scale = max(scale_h, scale_w)
                    
                    eh_end = int(eff_h * cover_scale)
                    ew_end = int(eff_w * cover_scale)
                    
                    # 初始大小
                    eh_start = 220
                    ew_start = int(eh_start * aspect_ratio)
                    
                    # 插值計算當前寬高
                    eh = int(eh_start + (eh_end - eh_start) * scale_ratio)
                    ew = int(ew_start + (ew_end - ew_start) * scale_ratio)
                    
                    dog_resized = cv.resize(dog_image, (ew, eh), interpolation=cv.INTER_LINEAR)
                    
                    # 棋盤格去背 (Chroma Keying)：轉換至 HSV，檢測並過濾灰白與純白網格背景
                    hsv_dog = cv.cvtColor(dog_resized, cv.COLOR_BGR2HSV)
                    s_dog = hsv_dog[:, :, 1]
                    v_dog = hsv_dog[:, :, 2]
                    # 飽和度低於 30 且 亮度高於 150 被判定為黑白/灰白相間格子
                    bg_mask = (s_dog < 30) & (v_dog > 150)
                    fg_mask = (~bg_mask).astype(np.uint8) * 255
                    
                    # 忍犬淡入比例 (1.0秒淡入)
                    alpha_blend = min(1.0, t_rel / 1.0)
                    
                    # 決定要貼上特效的所有中心點 (固定在畫面中央)
                    targets = [(w // 2, h // 2)]
                    
                    for cx, cy in targets:
                        x1 = max(0, cx - ew // 2)
                        y1 = max(0, cy - eh // 2)
                        x2 = min(w, cx + ew // 2)
                        y2 = min(h, cy + eh // 2)
                        
                        # 裁剪以防溢出邊界
                        crop_x1 = ew // 2 - (cx - x1)
                        crop_y1 = eh // 2 - (cy - y1)
                        crop_x2 = ew // 2 + (x2 - cx)
                        crop_y2 = eh // 2 + (y2 - cy)
                        
                        if (x2 > x1) and (y2 > y1):
                            roi = debug_image[y1:y2, x1:x2]
                            eff_cropped = dog_resized[crop_y1:crop_y2, crop_x1:crop_x2]
                            mask_cropped = fg_mask[crop_y1:crop_y2, crop_x1:crop_x2]
                            
                            # 結合淡入比例與去背 Mask
                            mask_f = mask_cropped.astype(np.float32) / 255.0
                            mask_f = np.expand_dims(mask_f * alpha_blend, axis=2)
                            
                            # 融合公式
                            blended = eff_cropped.astype(np.float32) * mask_f + roi.astype(np.float32) * (1.0 - mask_f)
                            debug_image[y1:y2, x1:x2] = blended.astype(np.uint8)
            else:
                is_playing_dog = False
                dog_image = None

        # 渲染口寄せの術 (MPKa7iiQyaoGxbGagC6Fea (1).png) - 固定於畫面中央並在 2 秒內縮放至佔滿整個視窗 (利用自帶 Alpha 通道精確去背，前 1 秒僅顯示煙霧)
        if is_playing_summon and summon_image is not None:
            dt_sum = time.time() - summon_start_time
            if dt_sum < 6.0:
                h, w = debug_image.shape[:2]
                
                # 前 1 秒僅顯示煙霧，1 秒後才開始淡入並放大召喚獸
                if dt_sum >= 1.0:
                    t_rel = dt_sum - 1.0
                    # 2 秒內從初始尺寸動態膨脹至佔滿整個視窗
                    scale_ratio = min(1.0, t_rel / 2.0)
                    
                    eff_h, eff_w = summon_image.shape[:2]
                    aspect_ratio = eff_w / eff_h
                    
                    # 計算 Cover 縮放：保持原始長寬比且長寬皆大於等於視窗大小
                    scale_h = h / eff_h
                    scale_w = w / eff_w
                    cover_scale = max(scale_h, scale_w)
                    
                    eh_end = int(eff_h * cover_scale)
                    ew_end = int(eff_w * cover_scale)
                    
                    # 初始大小
                    eh_start = 220
                    ew_start = int(eh_start * aspect_ratio)
                    
                    # 插值計算當前寬高
                    eh = int(eh_start + (eh_end - eh_start) * scale_ratio)
                    ew = int(ew_start + (ew_end - ew_start) * scale_ratio)
                    
                    summon_resized = cv.resize(summon_image, (ew, eh), interpolation=cv.INTER_LINEAR)
                    
                    # 自帶 Alpha 去背：第四個通道是前景遮罩，前三個通道是 BGR
                    fg_mask = summon_resized[:, :, 3]
                    summon_bgr = summon_resized[:, :, :3]
                    
                    # 淡入比例 (1.0秒淡入)
                    alpha_blend = min(1.0, t_rel / 1.0)
                    
                    # 決定要貼上特效的所有中心點 (固定在畫面中央)
                    targets = [(w // 2, h // 2)]
                    
                    for cx, cy in targets:
                        x1 = max(0, cx - ew // 2)
                        y1 = max(0, cy - eh // 2)
                        x2 = min(w, cx + ew // 2)
                        y2 = min(h, cy + eh // 2)
                        
                        # 裁剪以防溢出邊界
                        crop_x1 = ew // 2 - (cx - x1)
                        crop_y1 = eh // 2 - (cy - y1)
                        crop_x2 = ew // 2 + (x2 - cx)
                        crop_y2 = eh // 2 + (y2 - cy)
                        
                        if (x2 > x1) and (y2 > y1):
                            roi = debug_image[y1:y2, x1:x2]
                            eff_cropped = summon_bgr[crop_y1:crop_y2, crop_x1:crop_x2]
                            mask_cropped = fg_mask[crop_y1:crop_y2, crop_x1:crop_x2]
                            
                            # 結合淡入比例與去背 Mask
                            mask_f = mask_cropped.astype(np.float32) / 255.0
                            mask_f = np.expand_dims(mask_f * alpha_blend, axis=2)
                            
                            # 融合公式
                            blended = eff_cropped.astype(np.float32) * mask_f + roi.astype(np.float32) * (1.0 - mask_f)
                            debug_image[y1:y2, x1:x2] = blended.astype(np.uint8)
            else:
                is_playing_summon = False
                summon_image = None

        # 渲染千鳥雷電特效 (lightning-aura.gif) - 跟隨所有手部位置並隨時間動態放大至 250% (使用 Alpha 通道精確去背)
        if is_playing_chidori and chidori_reader is not None:
            dt = time.time() - chidori_start_time
            if dt < 5.0:
                ret_c, chidori_frame, chidori_alpha = chidori_reader.read()
                
                if ret_c:
                    h, w = debug_image.shape[:2]
                    
                    # 在 2 秒內快速動態放大 (從 1.0 放大到 2.5 倍)，之後維持最大尺寸
                    scale_ratio = min(1.0, dt / 2.0)
                    scale = 1.0 + 1.5 * scale_ratio
                    
                    # 基礎大小 300x300，最大放大到 750x750 (250%)
                    ew = int(300 * scale)
                    eh = int(300 * scale)
                    
                    chidori_resized = cv.resize(chidori_frame, (ew, eh))
                    alpha_resized = cv.resize(chidori_alpha, (ew, eh))
                    
                    # 決定要貼上特效的所有中心點
                    # 如果當前有偵測到合格的手部，則在所有手部中心貼上特效；若無，則使用最後記錄的位置
                    targets = current_hands if len(current_hands) > 0 else [(last_hand_cx, last_hand_cy)]
                    
                    for cx, cy in targets:
                        x1 = max(0, cx - ew // 2)
                        y1 = max(0, cy - eh // 2)
                        x2 = min(w, cx + ew // 2)
                        y2 = min(h, cy + eh // 2)
                        
                        # 裁剪以防溢出邊界
                        crop_x1 = ew // 2 - (cx - x1)
                        crop_y1 = eh // 2 - (cy - y1)
                        crop_x2 = ew // 2 + (x2 - cx)
                        crop_y2 = eh // 2 + (y2 - cy)
                        
                        if (x2 > x1) and (y2 > y1):
                            roi = debug_image[y1:y2, x1:x2]
                            eff_cropped = chidori_resized[crop_y1:crop_y2, crop_x1:crop_x2]
                            mask_cropped = alpha_resized[crop_y1:crop_y2, crop_x1:crop_x2]
                            
                            # 使用 alpha mask 進行融合去背，只疊加非透明(雷電)的部分
                            mask_f = mask_cropped.astype(np.float32) / 255.0
                            mask_f = np.expand_dims(mask_f, axis=2)
                            
                            # 融合公式：前景 * alpha + 背景 * (1 - alpha)
                            blended = eff_cropped.astype(np.float32) * mask_f + roi.astype(np.float32) * (1.0 - mask_f)
                            debug_image[y1:y2, x1:x2] = blended.astype(np.uint8)
            else:
                is_playing_chidori = False
                chidori_reader = None

        debug_image = draw_debug_image(
            debug_image,
            font_path,
            fps_result,
            labels,
            bboxes,
            scores,
            class_ids,
            score_th,
            erase_bbox,
            use_display_score,
            jutsu,
            sign_display_queue,
            sign_max_display,
            jutsu_display_time,
            jutsu_font_size_ratio,
            lang_offset,
            jutsu_index,
            jutsu_start_time,
        )
        if use_fullscreen:
            cv.setWindowProperty(window_name, cv.WND_PROP_FULLSCREEN,
                                 cv.WINDOW_FULLSCREEN)
        
        # 影像等比例縮放且多餘區域用黑色填充
        try:
            rect = cv.getWindowImageRect(window_name)
        except Exception:
            rect = None

        if rect is not None and rect[2] > 0 and rect[3] > 0:
            win_w, win_h = rect[2], rect[3]
            src_h, src_w = debug_image.shape[:2]
            
            aspect_ratio = src_w / src_h
            win_aspect = win_w / win_h
            
            if win_aspect > aspect_ratio:
                # 視窗太寬，以高度為基準
                new_h = win_h
                new_w = int(win_h * aspect_ratio)
            else:
                # 視窗太高，以寬度為基準
                new_w = win_w
                new_h = int(win_w / aspect_ratio)
            
            new_w = max(1, new_w)
            new_h = max(1, new_h)
            
            resized_image = cv.resize(debug_image, (new_w, new_h), interpolation=cv.INTER_AREA)
            
            # 建立黑色背景畫布
            canvas = np.zeros((win_h, win_w, 3), dtype=np.uint8)
            
            # 置中貼上影像
            x_offset = (win_w - new_w) // 2
            y_offset = (win_h - new_h) // 2
            canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized_image
            
            cv.imshow(window_name, canvas)
        else:
            cv.imshow(window_name, debug_image)

        # FPS調整 #############################################################
        elapsed_time = time.time() - start_time
        sleep_time = max(0, ((1.0 / fps) - elapsed_time))
        time.sleep(sleep_time)

    cap.release()
    cv.destroyAllWindows()


def check_jutsu(
    sign_history_queue,
    labels,
    jutsu,
    jutsu_index,
    jutsu_start_time,
):
    # 印の履歴から術名をマッチング
    sign_history = ''
    matched = False
    if len(sign_history_queue) > 0:
        for sign_id in sign_history_queue:
            sign_history = sign_history + labels[sign_id][1]
        for index, signs in enumerate(jutsu):
            if sign_history == ''.join(signs[4:]):
                jutsu_index = index
                jutsu_start_time = time.time()  # 術の最終検出時間
                matched = True
                break

    if matched:
        sign_history_queue.clear()

    return jutsu_index, jutsu_start_time, matched


def draw_debug_image(
    debug_image,
    font_path,
    fps_result,
    labels,
    bboxes,
    scores,
    class_ids,
    score_th,
    erase_bbox,
    use_display_score,
    jutsu,
    sign_display_queue,
    sign_max_display,
    jutsu_display_time,
    jutsu_font_size_ratio,
    lang_offset,
    jutsu_index,
    jutsu_start_time,
):
    frame_width, frame_height = debug_image.shape[1], debug_image.shape[0]

    # 印のバウンディングボックスの重畳表示(表示オプション有効時) ###################
    if not erase_bbox:
        for bbox, score, class_id in zip(bboxes, scores, class_ids):
            class_id = int(class_id) + 1

            # 検出閾値未満のバウンディングボックスは捨てる
            if score < score_th:
                continue

            x1, y1 = int(bbox[0]), int(bbox[1])
            x2, y2 = int(bbox[2]), int(bbox[3])

            # バウンディングボックス(長い辺にあわせて正方形を表示)
            x_len = x2 - x1
            y_len = y2 - y1
            square_len = x_len if x_len >= y_len else y_len
            square_x1 = int(((x1 + x2) / 2) - (square_len / 2))
            square_y1 = int(((y1 + y2) / 2) - (square_len / 2))
            square_x2 = square_x1 + square_len
            square_y2 = square_y1 + square_len
            cv.rectangle(debug_image, (square_x1, square_y1),
                         (square_x2, square_y2), (255, 255, 255), 4)
            cv.rectangle(debug_image, (square_x1, square_y1),
                         (square_x2, square_y2), (0, 0, 0), 2)

            # 印の種類
            font_size = int(square_len / 2)
            debug_image = CvDrawText.puttext(
                debug_image, labels[class_id][1],
                (square_x2 - font_size, square_y2 - font_size), font_path,
                font_size, (185, 0, 0))

            # 検出スコア(表示オプション有効時)
            if use_display_score:
                font_size = int(square_len / 8)
                debug_image = CvDrawText.puttext(
                    debug_image, '{:.3f}'.format(score),
                    (square_x1 + int(font_size / 4),
                     square_y1 + int(font_size / 4)), font_path, font_size,
                    (185, 0, 0))

    # ヘッダー作成：FPS #########################################################
    header_image = np.zeros((int(frame_height / 18), frame_width, 3), np.uint8)
    header_image = CvDrawText.puttext(header_image, "FPS:" + str(fps_result),
                                      (5, 0), font_path,
                                      int(frame_height / 20), (255, 255, 255))

    # フッター作成：印の履歴、および、術名表示 ####################################
    footer_image = np.zeros((int(frame_height / 10), frame_width, 3), np.uint8)

    # 印の履歴文字列生成
    sign_display = ''
    if len(sign_display_queue) > 0:
        for sign_id in sign_display_queue:
            sign_display = sign_display + labels[sign_id][1]

    # 術名表示(指定時間描画)
    if lang_offset == 0:
        separate_string = '・'
    else:
        separate_string = '：'
    if (time.time() - jutsu_start_time) < jutsu_display_time:
        if jutsu[jutsu_index][0] == '':  # 属性(火遁等)の定義が無い場合
            jutsu_string = jutsu[jutsu_index][2 + lang_offset]
        else:  # 属性(火遁等)の定義が有る場合
            jutsu_string = jutsu[jutsu_index][0 + lang_offset] + \
                separate_string + jutsu[jutsu_index][2 + lang_offset]
        footer_image = CvDrawText.puttext(
            footer_image, jutsu_string, (5, 0), font_path,
            int(frame_width / jutsu_font_size_ratio), (255, 255, 255))
    # 印表示
    else:
        footer_image = CvDrawText.puttext(footer_image, sign_display, (5, 0),
                                          font_path,
                                          int(frame_width / sign_max_display),
                                          (255, 255, 255))

    # ヘッダーとフッターをデバッグ画像へ結合 ######################################
    debug_image = cv.vconcat([header_image, debug_image])
    debug_image = cv.vconcat([debug_image, footer_image])

    return debug_image


if __name__ == '__main__':
    main()
