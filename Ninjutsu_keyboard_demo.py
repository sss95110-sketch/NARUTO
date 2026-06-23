#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import csv
import time
import copy
from collections import deque
import cv2 as cv
import numpy as np

from utils import CvDrawText

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sign_interval", type=float, default=2.0)
    parser.add_argument("--jutsu_display_time", type=int, default=5)
    parser.add_argument("--use_jutsu_lang_en", action="store_true")
    args = parser.parse_args()
    return args

def check_jutsu(sign_history_queue, labels, jutsu, jutsu_index, jutsu_start_time):
    sign_history = ''
    if len(sign_history_queue) > 0:
        for sign_id in sign_history_queue:
            sign_history = sign_history + labels[sign_id][1]
        for index, signs in enumerate(jutsu):
            if sign_history == ''.join(signs[4:]):
                jutsu_index = index
                jutsu_start_time = time.time()
                break
    return jutsu_index, jutsu_start_time

def draw_debug_image(
    debug_image, font_path, labels, jutsu, sign_display_queue,
    sign_max_display, jutsu_display_time, jutsu_font_size_ratio,
    lang_offset, jutsu_index, jutsu_start_time
):
    frame_width, frame_height = debug_image.shape[1], debug_image.shape[0]
    
    # 建立 Header
    header_image = np.zeros((int(frame_height / 18), frame_width, 3), np.uint8)
    header_image = CvDrawText.puttext(header_image, "KEYBOARD TEST MODE",
                                      (5, 0), font_path,
                                      int(frame_height / 20), (0, 255, 0))

    # 建立 Footer
    footer_image = np.zeros((int(frame_height / 10), frame_width, 3), np.uint8)
    
    # 生成結印歷史字串
    sign_display = ''
    if len(sign_display_queue) > 0:
        for sign_id in sign_display_queue:
            sign_display = sign_display + labels[sign_id][1]

    # 顯示術名 (指定時間內顯示)
    if lang_offset == 0:
        separate_string = '・'
    else:
        separate_string = '：'
        
    if (time.time() - jutsu_start_time) < jutsu_display_time:
        if jutsu[jutsu_index][0] == '':  # 無屬性定義
            jutsu_string = jutsu[jutsu_index][2 + lang_offset]
        else:  # 有屬性定義
            jutsu_string = jutsu[jutsu_index][0 + lang_offset] + \
                separate_string + jutsu[jutsu_index][2 + lang_offset]
        footer_image = CvDrawText.puttext(
            footer_image, jutsu_string, (5, 0), font_path,
            int(frame_width / jutsu_font_size_ratio), (0, 255, 0)) # 用綠色或白色顯示
    else:
        # 顯示印
        footer_image = CvDrawText.puttext(footer_image, sign_display, (5, 0),
                                          font_path,
                                          int(frame_width / sign_max_display),
                                          (255, 255, 255))

    # 拼接
    result_image = cv.vconcat([header_image, debug_image])
    result_image = cv.vconcat([result_image, footer_image])
    return result_image

def main():
    args = get_args()
    sign_interval = args.sign_interval
    jutsu_display_time = args.jutsu_display_time
    use_jutsu_lang_en = args.use_jutsu_lang_en

    font_path = './utils/font/衡山毛筆フォント.ttf'

    # 讀取 CSV
    with open('setting/labels.csv', encoding='utf8') as f:
        labels = csv.reader(f)
        labels = [row for row in labels]

    with open('setting/jutsu.csv', encoding='utf8') as f:
        jutsu = csv.reader(f)
        jutsu = [row for row in jutsu]

    sign_max_display = 18
    sign_max_history = 44
    sign_display_queue = deque(maxlen=sign_max_display)
    sign_history_queue = deque(maxlen=sign_max_history)

    lang_offset = 0
    jutsu_font_size_ratio = sign_max_display
    if use_jutsu_lang_en:
        lang_offset = 1
        jutsu_font_size_ratio = int((sign_max_display / 3) * 4)

    sign_interval_start = 0
    jutsu_index = 0
    jutsu_start_time = 0

    # 鍵盤映射表
    key_map = {
        ord('q'): 1,   # 子
        ord('w'): 2,   # 丑
        ord('e'): 3,   # 寅
        ord('r'): 4,   # 卯
        ord('t'): 5,   # 辰
        ord('y'): 6,   # 巳
        ord('u'): 7,   # 午
        ord('i'): 8,   # 未
        ord('o'): 9,   # 申
        ord('p'): 10,  # 酉
        ord('a'): 11,  # 戌
        ord('s'): 12,  # 亥
        ord('d'): 13,  # 祈 (合掌)
        ord('f'): 15,  # 壬
    }

    window_name = 'NARUTO HandSign Keyboard Test Demo'
    cv.namedWindow(window_name, cv.WINDOW_NORMAL)

    print("=== 火影忍者鍵盤結印測試模式 ===")
    print("點擊 OpenCV 視窗後，使用以下鍵盤按鍵來模擬結印：")
    print("q:子 w:丑 e:寅 r:卯 t:辰 y:巳 u:午")
    print("i:未 o:申 p:酉 a:戌 s:亥 d:合掌(祈) f:壬")
    print("c:清除結印歷史  |  ESC:退出")

    while True:
        # 建立底色畫布
        canvas = np.zeros((540, 960, 3), dtype=np.uint8)

        # 繪製操作指示
        canvas = CvDrawText.puttext(canvas, "【鍵盤結印測試模式】點擊此視窗後，按鍵即可模擬結印！", (30, 40), font_path, 24, (0, 255, 255))
        
        # 繪製按鍵指引
        canvas = CvDrawText.puttext(canvas, "按鍵對照：", (30, 110), font_path, 20, (255, 255, 255))
        canvas = CvDrawText.puttext(canvas, "  q: 子 (Rat)     w: 丑 (Ox)      e: 寅 (Tiger)    r: 卯 (Hare)", (30, 150), font_path, 18, (180, 180, 180))
        canvas = CvDrawText.puttext(canvas, "  t: 辰 (Dragon)  y: 巳 (Snake)   u: 午 (Horse)    i: 未 (Ram)", (30, 190), font_path, 18, (180, 180, 180))
        canvas = CvDrawText.puttext(canvas, "  o: 申 (Monkey)  p: 酉 (Bird)     a: 戌 (Dog)      s: 亥 (Boar)", (30, 230), font_path, 18, (180, 180, 180))
        canvas = CvDrawText.puttext(canvas, "  d: 祈 (合掌)    f: 壬 (Mizunoe)", (30, 270), font_path, 18, (180, 180, 180))

        canvas = CvDrawText.puttext(canvas, "功能鍵：", (30, 330), font_path, 20, (255, 255, 255))
        canvas = CvDrawText.puttext(canvas, "  c: 清除目前的結印歷史紀錄", (30, 370), font_path, 18, (180, 180, 180))
        canvas = CvDrawText.puttext(canvas, "  ESC: 離開程式", (30, 410), font_path, 18, (180, 180, 180))

        # 如果前一個印的間隔超時，清空
        if (time.time() - sign_interval_start) > sign_interval:
            sign_display_queue.clear()
            sign_history_queue.clear()

        # 繪製 Header & Footer 並合成最終圖片
        debug_image = draw_debug_image(
            canvas,
            font_path,
            labels,
            jutsu,
            sign_display_queue,
            sign_max_display,
            jutsu_display_time,
            jutsu_font_size_ratio,
            lang_offset,
            jutsu_index,
            jutsu_start_time
        )

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

        # 偵測鍵盤按鍵，最長等待 50 毫秒（降低 CPU 佔用）
        key = cv.waitKey(50)

        # 處理按鍵
        if key == 27:  # ESC
            break
        elif key == 99 or key == 67:  # C / c
            sign_display_queue.clear()
            sign_history_queue.clear()
            print("已清除結印紀錄")
        elif key in key_map:
            class_id = key_map[key]
            
            # 只有當隊列為空，或是與上一個印不同時才加入（避免長按連續輸入）
            if len(sign_display_queue) == 0 or sign_display_queue[-1] != class_id:
                sign_display_queue.append(class_id)
                sign_history_queue.append(class_id)
                sign_interval_start = time.time()
                print(f"輸入印：{labels[class_id][1]}")

                # 進行術判定
                jutsu_index, jutsu_start_time = check_jutsu(
                    sign_history_queue,
                    labels,
                    jutsu,
                    jutsu_index,
                    jutsu_start_time
                )

    cv.destroyAllWindows()

if __name__ == '__main__':
    main()
