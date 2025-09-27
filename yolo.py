# -*- coding: utf-8 -*-
import time
import threading
from pynput import keyboard
import threading
from threading import Event

def start_yolo_follow_optimized(target_class='truck', model_name='yolov8s.pt', 
                             exit_key=keyboard.Key.esc, check_interval=0.05, confidence=0.4):
    """
    坦克识别与追踪系统
    参数:
        target_class: 要检测的目标类别，默认为'truck'
        model_name: YOLO模型路径，默认为'yolov8s.pt'
        exit_key: 退出按键，默认为ESC键
        check_interval: 检测间隔(秒)
        confidence: 置信度阈值
    """
    from ultralytics import YOLO
    import cv2
    import numpy as np
    import mss
    import pyautogui
    import time
    import os
    
    stop_flag = {'stop': False}
    performance_stats = {'fps': 0, 'detection_count': 0}
    
    # 键盘监听回调
    def on_press(key):
        if key == exit_key:
            print("\n检测到退出键，正在停止...")
            stop_flag['stop'] = True
            return False  # 停止监听
    
    # 启动键盘监听线程
    listener = keyboard.Listener(on_press=on_press)
    listener.start()
    
    def worker():
        print("正在初始化YOLO模型...")
        try:
            # 加载模型
            model = YOLO(model_name)
            print("模型加载成功: {}".format(model_name))
            
            # 打印可用类别
            class_names = model.names
            print("\n模型支持的类别:")
            for i, name in class_names.items():
                print("  {}: {}".format(i, name))
            
            # 查找最匹配的类别
            class_id = None
            target_lower = target_class.lower()
            
            # 先尝试完全匹配
            for k, v in class_names.items():
                if v.lower() == target_lower:
                    class_id = k
                    break
            
            # 如果没有完全匹配，尝试部分匹配
            if class_id is None:
                for k, v in class_names.items():
                    if target_lower in v.lower() or v.lower() in target_lower:
                        class_id = k
                        print("警告: 未找到完全匹配的类别，使用最接近的类别:")
                        break
            
            if class_id is None:
                print("提示: 可以尝试使用 'truck' 或 'car' 作为目标类别")
                return
            
            print("置信度阈值: {}".format(confidence))
            print("\n正在启动检测...")
            print("按 ESC 键退出程序")
            
            # 模型预热
            print("正在预热模型...")
            model(np.zeros((640, 640, 3), dtype=np.uint8))  # 预热推理
            
            # 屏幕设置
            monitor = mss.mss().monitors[1]
            screen_region = {
                'left': monitor['left'],
                'top': monitor['top'],
                'width': monitor['width'],
                'height': monitor['height']
            }
            
            # 性能优化
            frame_count = 0
            start_time = time.time()
            last_target = None
            
            with mss.mss() as sct:
                while not stop_flag['stop']:
                    
                    try:
                        # 性能计时
                        frame_start = time.time()
                        
                        # 捕获屏幕
                        screenshot = sct.grab(screen_region)
                        img = np.array(screenshot)[:, :, :3]
                        
                        # YOLO检测
                        results = model(img, conf=confidence, verbose=False)
                        
                        # 处理检测结果
                        best_target = None
                        max_area = 0
                        
                        if results[0].boxes is not None:
                            for box in results[0].boxes:
                                if int(box.cls) == class_id and box.conf > confidence:
                                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                                    area = (x2 - x1) * (y2 - y1)
                                    
                                    # 选择最大的目标
                                    if area > max_area:
                                        max_area = area
                                        best_target = (x1, y1, x2, y2, box.conf)
                        
                        # 移动鼠标到目标中心
                        if best_target:
                            x1, y1, x2, y2, conf = best_target
                            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                            pyautogui.moveTo(cx, cy, duration=0.1)
                            print("检测到目标: 位置({}, {}), 置信度: {:.2f}".format(cx, cy, conf))
                            performance_stats['detection_count'] += 1
                            last_target = (cx, cy)
                        
                        # 性能统计
                        frame_count += 1
                        if time.time() - start_time > 1.0:  # 每秒更新FPS
                            performance_stats['fps'] = frame_count / (time.time() - start_time)
                            print("FPS: {:.1f}, 检测次数: {}".format(performance_stats['fps'], performance_stats['detection_count']))
                            frame_count = 0
                            start_time = time.time()
                        
                        # 控制检测频率
                        time.sleep(check_interval)
                        
                    except Exception as e:
                        print("检测过程中出错: {}".format(str(e)))
                        time.sleep(1)  # 出错时暂停1秒
                        continue
                        
        except Exception as e:
            print("初始化YOLO模型时发生错误: {}".format(str(e)))
            print("请检查模型文件和依赖项是否正确安装")
            return
        
        # 主循环已在上面的try块中实现
        print("程序结束")
        return
                
    # 启动工作线程
    t = threading.Thread(target=worker)
    t.daemon = True
    t.start()
    
    try:
        while t.is_alive():
            t.join(0.1)
    except KeyboardInterrupt:
        stop_flag['stop'] = True
        t.join()
    
    print("程序已退出")
    return stop_flag, performance_stats



def main():
    """完整的YOLO追踪示例"""
    
    print("🚀 启动YOLO目标追踪...")
    
    # 启动追踪
    stop_flag, performance_stats = start_yolo_follow_optimized(
        target_class='Tank',          # 可以改为 'person', 'dog', 'cell phone' 等
        model_name='best.pt',     # 模型选择：n=纳米，s=小，m=中，l=大，x=超大
        exit_key='q',                # 按Q键退出
        check_interval=0.02,         # 检测间隔20ms
        confidence=0.5               # 置信度阈值50%
    )
    
    try:
        # 主线程可以继续做其他事情
        while not stop_flag['stop']:
            # 实时显示性能统计
            #print(f"\rFPS: {performance_stats['fps']:.1f} | 检测次数: {performance_stats['detection_count']} | 按Q退出", end='')
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\n👋 程序被中断")
    finally:
        # 确保停止追踪
        stop_flag['stop'] = True
        print("\n✅ 追踪已停止")

if __name__ == "__main__":
    main()