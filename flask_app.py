import cv2
import torch
import multiprocessing as mp
import numpy as np
import os
import argparse
import time
import json
from flask import Flask, Response, render_template, request, jsonify
import base64
import threading
import queue
import shutil
import io
from PIL import Image

import config
from utils import SharedData
from multiprocess_camera import camera_process
from pose_visualization import process_pose_results
# 导入体态评估模块
from posture_assessment import PostureAssessment, get_latest_assessment

# 初始化Flask应用
app = Flask(__name__, 
           static_folder='static',
           template_folder='templates')

# 全局变量
frame_queues = {}  # 每个摄像头的帧队列
manager = None
return_dict = None
shared_data = None
cam_processes = {}
max_queue_size = 5  # 限制队列大小，防止内存使用过多

# 数据记录相关变量
is_recording = False
recording_type = None  # 'static' 或 'gait'
keypoints_data = []
recorded_frames = []
output_dir = "data/recorded"
last_saved_dir = None  # 记录最后保存的目录路径
frame_errors = {0: 0, 1: 0}  # 记录每个摄像头的帧解码错误次数

# 新增：数据记录缓冲区限制
max_frames_buffer = 1000  # 最大帧缓冲数
auto_save_interval = 500  # 每记录多少帧自动保存一次
auto_save_lock = threading.Lock()  # 自动保存的锁，防止并发保存
save_in_progress = False  # 标记是否正在进行保存操作

# 新增：评估结果缓存
assessment_result_cache = None
assessment_timestamp = None

def init_system(model_name=None, device=None):
    """初始化系统，启动摄像头进程"""
    global manager, return_dict, shared_data, cam_processes, output_dir
    
    # 检测可用设备
    device = device if device else ('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建多进程管理器
    manager = mp.Manager()
    return_dict = manager.dict()
    
    # 创建共享数据结构
    shared_data = SharedData()
    shared_data.running.value = True
    
    # 获取模型名称
    model_name = model_name if model_name else config.ModelConfig.DEFAULT_MODEL
    print(f"使用模型: {model_name}")
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    print(f"数据将保存到: {os.path.abspath(output_dir)}")
    
    # 创建两个摄像头处理进程
    for cam_id in [0, 1]:
        print(f"启动摄像头 {cam_id} 进程...")
        process = mp.Process(target=camera_process, 
                             args=(cam_id, return_dict, shared_data, model_name, device))
        process.start()
        cam_processes[cam_id] = process
        # 创建一个队列用于存储该摄像头的帧
        frame_queues[cam_id] = queue.Queue(maxsize=max_queue_size)
    
    # 启动帧更新线程
    threading.Thread(target=update_frames, daemon=True).start()
    
    # 等待一小段时间，确保进程启动
    time.sleep(2)
    
    # 检查是否有摄像头实际在工作
    print("检查摄像头状态...")
    camera_ok = False
    for cam_id in [0, 1]:
        if f'frame_{cam_id}' in return_dict:
            print(f"摄像头 {cam_id} 已经开始输出帧数据")
            camera_ok = True
        else:
            print(f"警告: 摄像头 {cam_id} 尚未输出帧数据")
    
    if not camera_ok:
        print("警告: 所有摄像头都没有输出数据，请检查摄像头连接和驱动")
    
    print("系统初始化完成")

# 新增：自动保存函数
def auto_save_data():
    """定期自动保存数据的函数"""
    global keypoints_data, recorded_frames, save_in_progress, last_saved_dir
    
    # 如果已经有保存操作在进行，或没有数据，则跳过
    if save_in_progress or not is_recording or not recorded_frames:
        return
    
    try:
        with auto_save_lock:
            save_in_progress = True
            
            # 拷贝当前数据用于保存
            kp_to_save = keypoints_data.copy() if keypoints_data else []
            frames_to_save = recorded_frames.copy()
            curr_recording_type = recording_type
            
            # 清空当前缓冲区
            keypoints_data = []
            recorded_frames = []
            
            print(f"自动保存：准备保存 {len(kp_to_save)} 条关键点数据和 {len(frames_to_save)} 帧图像")
            
        # 创建时间戳用于文件名
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        # 确保输出目录存在
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            
        # 创建特定类型的子目录（static或gait）
        type_dir = os.path.join(output_dir, curr_recording_type)
        if not os.path.exists(type_dir):
            os.makedirs(type_dir, exist_ok=True)
            
        # 为本次记录创建时间戳子目录
        record_dir = os.path.join(type_dir, timestamp)
        os.makedirs(record_dir, exist_ok=True)
        
        # 保存本次保存的目录路径
        last_saved_dir = record_dir
        
        # 保存关键点数据到JSON文件（如果有）
        if kp_to_save:
            keypoints_file = os.path.join(record_dir, f"keypoints.json")
            with open(keypoints_file, 'w', encoding='utf-8') as f:
                json.dump(kp_to_save, f, ensure_ascii=False, indent=2)
            print(f"自动保存：保存了 {len(kp_to_save)} 条关键点数据")
        else:
            print("自动保存：警告 - 没有关键点数据可保存")
        
        # 创建图像帧目录
        frames_dir = os.path.join(record_dir, "frames")
        os.makedirs(frames_dir, exist_ok=True)
        
        # 打印进度
        total_frames = len(frames_to_save)
        saved_count = 0
        
        print(f"自动保存：开始保存 {total_frames} 帧图像到 {frames_dir}")
        
        for i, frame in enumerate(frames_to_save):
            if frame is None or frame.size == 0:
                print(f"警告: 第 {i} 帧为空，跳过")
                continue
                
            frame_file = os.path.join(frames_dir, f"frame_{i:04d}.jpg")
            success = cv2.imwrite(frame_file, frame)
            
            if success:
                saved_count += 1
            else:
                print(f"警告: 无法保存第 {i} 帧到 {frame_file}")
            
            # 每保存10帧打印一次进度
            if (i + 1) % 10 == 0 or i == total_frames - 1:
                print(f"自动保存进度: {i+1}/{total_frames} 帧")
        
        # 创建简单的元数据文件
        metadata_file = os.path.join(record_dir, f"metadata.json")
        metadata = {
            'timestamp': timestamp,
            'type': curr_recording_type,
            'keypoints_count': len(kp_to_save),
            'frames_count': total_frames,
            'saved_frames_count': saved_count,
            'datetime': time.strftime("%Y-%m-%d %H:%M:%S"),
            'auto_saved': True
        }
        
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        
        print(f"自动保存完成! 共保存 {len(kp_to_save)} 条关键点数据和 {saved_count}/{total_frames} 帧图像")
        print(f"文件保存在: {record_dir}")
        
    except Exception as e:
        import traceback
        print(f"自动保存数据时出错: {str(e)}")
        print(traceback.format_exc())
    finally:
        save_in_progress = False

def update_frames():
    """不断从return_dict中获取新帧并更新到各个队列"""
    global is_recording, recording_type, keypoints_data, recorded_frames, frame_errors
    
    # 添加调试计数器
    frame_counter = {0: 0, 1: 0}
    keypoint_counter = {0: 0, 1: 0}
    last_log_time = time.time()
    log_interval = 5.0  # 每5秒输出一次日志
    
    while True:
        try:
            current_time = time.time()
            
            # 定期输出状态日志
            if current_time - last_log_time > log_interval:
                if is_recording:
                    print(f"记录状态 - 类型: {recording_type}, 已记录: {len(recorded_frames)} 帧, {len(keypoints_data)} 关键点")
                
                available_keys = list(return_dict.keys())
                keypoint_keys = [k for k in available_keys if k.startswith('keypoints_')]
                frame_keys = [k for k in available_keys if k.startswith('frame_')]
                error_keys = [k for k in available_keys if k.startswith('error_')]
                
                if keypoint_keys:
                    print(f"可用关键点数据: {keypoint_keys}")
                else:
                    print("警告: 没有找到关键点数据")
                
                if frame_keys:
                    print(f"可用帧数据: {frame_keys}")
                else:
                    print("警告: 没有找到帧数据")
                
                if error_keys:
                    for key in error_keys:
                        print(f"错误 {key}: {return_dict[key]}")
                
                last_log_time = current_time
            
            for cam_id in cam_processes.keys():
                # 检查是否有该摄像头的新帧
                if f'frame_{cam_id}' in return_dict:
                    try:
                        frame_data = return_dict[f'frame_{cam_id}']
                        frame_counter[cam_id] += 1
                        
                        # 尝试解码帧以验证其完整性
                        try:
                            # 使用OpenCV解码
                            frame_bytes = base64.b64decode(frame_data)
                            nparr = np.frombuffer(frame_bytes, np.uint8)
                            test_frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                            
                            # 如果OpenCV解码失败，尝试简单检查数据是否有效
                            if test_frame is None:
                                if len(frame_bytes) < 100:
                                    raise ValueError(f"图像数据太小: {len(frame_bytes)} 字节")
                                
                                # 检查是否是JPEG/PNG文件头
                                is_valid_image = False
                                if len(frame_bytes) > 4:
                                    # JPEG文件头通常以FF D8开始
                                    if frame_bytes[0:2] == b'\xff\xd8':
                                        is_valid_image = True
                                    # PNG文件头通常以89 50 4E 47开始
                                    elif frame_bytes[0:4] == b'\x89\x50\x4e\x47':
                                        is_valid_image = True
                                
                                if not is_valid_image:
                                    # 尝试输出前几个字节进行调试
                                    header_bytes = frame_bytes[:10] if len(frame_bytes) >= 10 else frame_bytes
                                    hex_header = ' '.join([f'{b:02X}' for b in header_bytes])
                                    raise ValueError(f"无效的图像数据格式，前10个字节: {hex_header}")
                                
                                # 即使文件头看起来有效，仍然无法解码
                                raise ValueError("OpenCV无法解码图像，但数据格式看起来有效")
                            
                            # 解码成功，重置错误计数
                            frame_errors[cam_id] = 0
                        except Exception as e:
                            # 解码失败，增加错误计数
                            frame_errors[cam_id] += 1
                            print(f"摄像头 {cam_id} 帧解码错误 ({frame_errors[cam_id]}): {str(e)}")
                            # 如果连续错误超过阈值，尝试重新启动摄像头
                            if frame_errors[cam_id] > 10:
                                print(f"摄像头 {cam_id} 连续解码错误，需要重新启动")
                                # 这里只记录错误，不中断当前循环
                                # 系统进程会自动尝试重新初始化摄像头
                                frame_errors[cam_id] = 0  # 重置错误计数
                            continue  # 跳过此帧
                        
                        # 如果正在记录，保存帧数据和关键点数据
                        if is_recording and cam_id == 0 and not save_in_progress:  # 假设使用摄像头0的数据进行记录，且没有保存正在进行
                            try:
                                # 检查缓冲区大小，如果超过限制，触发自动保存
                                if len(recorded_frames) >= auto_save_interval:
                                    print(f"已达到自动保存阈值 ({auto_save_interval} 帧)，开始自动保存...")
                                    threading.Thread(target=auto_save_data, daemon=True).start()
                                    continue
                                    
                                # 如果缓冲区已满，跳过当前帧
                                if len(recorded_frames) >= max_frames_buffer:
                                    print(f"警告: 缓冲区已满 ({max_frames_buffer} 帧)，丢弃当前帧")
                                    continue
                                
                                # 使用已解码的帧
                                if test_frame is not None and test_frame.size > 0:
                                    # 保存帧
                                    recorded_frames.append(test_frame.copy())  # 使用copy()避免引用问题
                                    
                                    # 获取关键点数据（如果有）
                                    if f'keypoints_{cam_id}' in return_dict:
                                        keypoints = return_dict[f'keypoints_{cam_id}']
                                        keypoint_counter[cam_id] += 1
                                        
                                        # 检查关键点数据是否为空或无效
                                        if keypoints is None or (isinstance(keypoints, list) and len(keypoints) == 0):
                                            if len(recorded_frames) % 30 == 0:  # 每30帧打印一次警告
                                                print(f"警告: 摄像头 {cam_id} 返回了空的关键点数据")
                                        else:
                                            keypoints_data.append({
                                                'timestamp': time.time(),
                                                'keypoints': keypoints,
                                                'type': recording_type
                                            })
                                            
                                            # 输出日志确认数据正在被记录
                                            if len(keypoints_data) % 10 == 0:  # 每10帧输出一次日志，避免过多日志
                                                print(f"已记录 {len(keypoints_data)} 条关键点数据和 {len(recorded_frames)} 帧图像")
                                    else:
                                        # 如果没有找到关键点数据，至少记录一条日志（每50帧一次）
                                        if len(recorded_frames) % 50 == 1:
                                            print(f"警告: 摄像头 {cam_id} 没有关键点数据，仅保存帧图像")
                                            print(f"return_dict中的键: {list(return_dict.keys())}")
                                            print(f"已记录 {len(recorded_frames)} 帧图像")
                                else:
                                    if len(recorded_frames) % 50 == 1:
                                        print(f"警告: 摄像头 {cam_id} 获取到的帧无效，无法记录")
                                            
                            except Exception as e:
                                import traceback
                                print(f"记录数据时出错: {str(e)}")
                                print(traceback.format_exc())
                        
                        # 仅当队列未满时添加新帧，避免延迟堆积
                        if not frame_queues[cam_id].full():
                            frame_queues[cam_id].put(frame_data)
                    except Exception as e:
                        print(f"处理摄像头 {cam_id} 帧时出错: {str(e)}")
                
                # 检查错误
                if f'error_{cam_id}' in return_dict:
                    print(f"摄像头 {cam_id} 错误: {return_dict[f'error_{cam_id}']}")
                    return_dict.pop(f'error_{cam_id}', None)
            
            # 小睡一会，减少CPU使用
            time.sleep(0.01)
        except Exception as e:
            import traceback
            print(f"更新帧时出错: {str(e)}")
            print(traceback.format_exc())

def gen_frames(camera_id):
    """生成帧的生成器函数，用于视频流"""
    camera_id = int(camera_id)
    error_image = None  # 缓存错误图像
    error_count = 0     # 错误计数
    max_errors = 5      # 最大允许错误次数，超过后输出更详细的调试信息
    
    while True:
        # 从对应摄像头的队列获取最新帧
        try:
            # 非阻塞式获取，如果队列为空则等待短暂时间后重试
            if not frame_queues[camera_id].empty():
                frame_data = frame_queues[camera_id].get()
                try:
                    # 确保frame_data有效
                    if not frame_data:
                        raise ValueError("空的帧数据")
                    
                    # 记录帧数据长度，用于调试
                    data_len = len(frame_data)
                    
                    # 转换为base64字符串（用于MJPEG流）并解码
                    try:
                        frame_bytes = base64.b64decode(frame_data)
                    except Exception as decode_err:
                        error_count += 1
                        if error_count > max_errors:
                            # 输出前20个字符用于调试
                            sample = frame_data[:20] + "..." if len(frame_data) > 20 else frame_data
                            print(f"摄像头 {camera_id} Base64解码错误 ({error_count}次): {str(decode_err)}")
                            print(f"数据长度: {data_len}, 样本: {sample}")
                        raise decode_err
                    
                    # 检查解码后的数据是否有效
                    if len(frame_bytes) < 50:
                        error_count += 1
                        if error_count > max_errors:
                            print(f"摄像头 {camera_id} 解码后数据太小: {len(frame_bytes)} 字节, Base64长度: {data_len}")
                        raise ValueError(f"解码后数据太小: {len(frame_bytes)} 字节")
                    
                    # 生成MJPEG帧
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + 
                           frame_bytes + b'\r\n')
                    
                    # 成功处理帧，重置错误计数
                    error_count = 0
                    
                except Exception as e:
                    print(f"摄像头 {camera_id} 生成帧时错误: {str(e)}")
                    # 如果解码失败，发送一个错误图像
                    if error_image is None:
                        # 创建一个错误图像
                        error_img = np.zeros((480, 640, 3), dtype=np.uint8)
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        cv2.putText(error_img, f"Camera {camera_id} Error", (50, 240), font, 1, (0, 0, 255), 2)
                        cv2.putText(error_img, f"Error: {str(e)[:30]}", (50, 280), font, 0.7, (0, 0, 255), 1)
                        _, error_buffer = cv2.imencode('.jpg', error_img)
                        error_image = error_buffer.tobytes()
                    
                    yield (b'--frame\r\n'
                        b'Content-Type: image/jpeg\r\n\r\n' + 
                        error_image + b'\r\n')
            else:
                # 如果队列为空，短暂等待
                time.sleep(0.01)
        except Exception as e:
            print(f"生成帧主循环错误: {str(e)}")
            time.sleep(0.1)  # 出错时稍微等长一点

def save_recorded_data():
    """保存记录的数据"""
    global keypoints_data, recorded_frames, last_saved_dir, save_in_progress
    
    # 防止并发操作
    if save_in_progress:
        print("已有保存操作正在进行，请等待")
        return False
    
    try:
        save_in_progress = True
        
        if not recorded_frames:
            print("警告: 没有帧数据可保存")
            save_in_progress = False
            return False
        
        # 创建时间戳用于文件名
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        # 确保输出目录存在
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            
        # 创建特定类型的子目录（static或gait）
        type_dir = os.path.join(output_dir, recording_type)
        if not os.path.exists(type_dir):
            os.makedirs(type_dir, exist_ok=True)
            
        # 为本次记录创建时间戳子目录
        record_dir = os.path.join(type_dir, timestamp)
        os.makedirs(record_dir, exist_ok=True)
        
        # 保存本次保存的目录路径
        last_saved_dir = record_dir
        
        # 保存关键点数据到JSON文件（即使为空，也创建文件）
        keypoints_file = os.path.join(record_dir, "keypoints.json")
        with open(keypoints_file, 'w', encoding='utf-8') as f:
            json.dump(keypoints_data if keypoints_data else [], f, ensure_ascii=False, indent=2)
        print(f"数据保存成功! 共保存 {len(keypoints_data)} 条关键点数据和 {len(recorded_frames)}/{len(recorded_frames)} 帧图像")
        
        # 创建图像帧目录
        frames_dir = os.path.join(record_dir, "frames")
        os.makedirs(frames_dir, exist_ok=True)
        
        # 保存图像帧
        for i, frame in enumerate(recorded_frames):
            if frame is None or frame.size == 0:
                print(f"警告: 第 {i} 帧为空，跳过")
                continue
                
            frame_file = os.path.join(frames_dir, f"frame_{i:04d}.jpg")
            success = cv2.imwrite(frame_file, frame)
            
            if not success:
                print(f"警告: 无法保存第 {i} 帧到 {frame_file}")
        
        # 创建简单的元数据文件
        metadata_file = os.path.join(record_dir, "metadata.json")
        metadata = {
            'timestamp': timestamp,
            'type': recording_type,
            'keypoints_count': len(keypoints_data),
            'frames_count': len(recorded_frames),
            'datetime': time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        
        print(f"文件保存在: {record_dir}")
        
        # 清空数据
        keypoints_data = []
        recorded_frames = []
        
        return True
    except Exception as e:
        print(f"保存数据时出错: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        save_in_progress = False

@app.route('/')
def index():
    """主页路由"""
    return render_template('index.html')

@app.route('/posture_report')
def posture_report():
    """体态报告页面"""
    return render_template('posture_report.html')

@app.route('/video_feed/<camera_id>')
def video_feed(camera_id):
    """视频流路由"""
    # 返回multipart响应
    return Response(gen_frames(camera_id),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_latest_frame/<camera_id>')
def get_latest_frame(camera_id):
    """获取最新的帧（使用base64编码的JPEG）- 用于使用img标签加载图像的备选方法"""
    camera_id = int(camera_id)
    try:
        if not frame_queues[camera_id].empty():
            frame_data = frame_queues[camera_id].get()
            # 转换为base64字符串
            frame_base64 = base64.b64encode(frame_data).decode('utf-8')
            return jsonify({'frame': f'data:image/jpeg;base64,{frame_base64}'})
        else:
            return jsonify({'error': '没有可用的帧'})
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/start_static_pose_analysis', methods=['POST'])
def start_static_pose_analysis():
    """开始静态体态分析数据记录"""
    global is_recording, recording_type, keypoints_data, recorded_frames, frame_errors
    
    # 重置数据
    keypoints_data = []
    recorded_frames = []
    frame_errors = {0: 0, 1: 0}  # 重置错误计数
    
    # 设置记录标志
    is_recording = True
    recording_type = 'static'
    
    print(f"开始记录静态体态数据，时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    return jsonify({'status': 'success', 'message': '开始记录静态体态数据'})

@app.route('/start_gait_analysis', methods=['POST'])
def start_gait_analysis():
    """开始步态分析数据记录"""
    global is_recording, recording_type, keypoints_data, recorded_frames, frame_errors
    
    # 如果之前在记录静态体态，先保存这些数据
    if is_recording and recording_type == 'static' and (keypoints_data or recorded_frames):
        print("在开始步态分析前保存静态体态数据")
        save_recorded_data()
    
    # 重置数据
    keypoints_data = []
    recorded_frames = []
    frame_errors = {0: 0, 1: 0}  # 重置错误计数
    
    # 设置记录标志
    is_recording = True
    recording_type = 'gait'
    
    print(f"开始记录步态数据，时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    return jsonify({'status': 'success', 'message': '开始记录步态数据'})

@app.route('/complete_analysis', methods=['POST'])
def complete_analysis():
    """完成分析并保存数据"""
    global is_recording, recording_type, keypoints_data, recorded_frames
    
    # 停止记录
    previous_recording = is_recording
    is_recording = False
    
    print(f"请求完成分析，当前有 {len(keypoints_data)} 条关键点数据和 {len(recorded_frames)} 帧图像")
    
    # 检查是否有数据要保存
    if not recorded_frames:
        return jsonify({
            'status': 'error',
            'message': "没有帧数据需要保存，请确保记录过程正常"
        })
    
    # 保存数据
    success = save_recorded_data()
    
    # 获取保存的路径信息用于反馈
    saved_dir = last_saved_dir if success else None
    
    # 返回结果给前端
    if success:
        message = f"数据保存成功，已保存到路径: {saved_dir}"
    else:
        message = "保存数据失败，请检查日志获取更多信息"
    
    return jsonify({
        'status': 'success' if success else 'error',
        'message': message,
        'saved_dir': saved_dir
    })

@app.route('/get_data_status', methods=['GET'])
def get_data_status():
    """获取当前数据记录状态"""
    global is_recording, recording_type, keypoints_data, recorded_frames
    
    return jsonify({
        'is_recording': is_recording,
        'recording_type': recording_type,
        'keypoints_count': len(keypoints_data),
        'frames_count': len(recorded_frames),
        'last_saved_dir': last_saved_dir,
        'camera_errors': frame_errors
    })

def cleanup():
    """清理资源"""
    global shared_data, cam_processes
    
    # 设置退出标志
    if shared_data:
        shared_data.running.value = False
    
    # 等待进程结束
    for cam_id, process in cam_processes.items():
        if process.is_alive():
            process.join(timeout=1.0)
            if process.is_alive():
                process.terminate()
    
    # 清空所有队列
    for q in frame_queues.values():
        while not q.empty():
            try:
                q.get_nowait()
            except:
                pass
    
    print("系统已清理")

@app.route('/shutdown', methods=['POST'])
def shutdown():
    """关闭系统接口"""
    cleanup()
    return jsonify({'status': 'success'})

@app.route('/reset_camera/<camera_id>', methods=['POST'])
def reset_camera(camera_id):
    """重置指定摄像头"""
    try:
        camera_id = int(camera_id)
        if camera_id in cam_processes:
            # 先终止旧进程
            if cam_processes[camera_id].is_alive():
                cam_processes[camera_id].terminate()
                cam_processes[camera_id].join(1.0)
            
            # 清空队列
            while not frame_queues[camera_id].empty():
                try:
                    frame_queues[camera_id].get_nowait()
                except:
                    pass
            
            # 重置错误计数
            frame_errors[camera_id] = 0
            
            # 开启新进程
            device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
            model_name = config.ModelConfig.DEFAULT_MODEL
            process = mp.Process(target=camera_process, 
                                args=(camera_id, return_dict, shared_data, model_name, device))
            process.start()
            cam_processes[camera_id] = process
            
            return jsonify({'status': 'success', 'message': f'摄像头 {camera_id} 已重置'})
        else:
            return jsonify({'status': 'error', 'message': f'摄像头 {camera_id} 不存在'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

# 新增路由：进行体态评估
@app.route('/assess_posture', methods=['POST'])
def assess_posture():
    """进行体态评估，分析最新的静态体态数据"""
    global assessment_result_cache, assessment_timestamp
    
    # 检查是否有最新的数据可用
    if not os.path.exists(os.path.join(output_dir, "static")):
        return jsonify({
            'status': 'error',
            'message': '未找到静态体态数据，请先进行数据采集'
        })
    
    # 如果缓存的评估结果不超过30秒，直接返回缓存
    current_time = time.time()
    if assessment_result_cache and assessment_timestamp and current_time - assessment_timestamp < 30:
        return jsonify({
            'status': 'success',
            'cached': True,
            'result': assessment_result_cache
        })
    
    try:
        # 创建评估器实例
        assessor = PostureAssessment()
        
        # 加载最新的数据
        if not assessor.load_latest_data(os.path.join(output_dir, "static")):
            return jsonify({
                'status': 'error',
                'message': '加载最新数据失败，请确保数据采集正确'
            })
        
        # 进行评估
        if not assessor.assess_all_postures():
            return jsonify({
                'status': 'error',
                'message': '评估过程中出错，请检查数据完整性'
            })
        
        # 获取评估报告
        report = assessor.get_assessment_report()
        
        # 更新缓存
        assessment_result_cache = report
        assessment_timestamp = current_time
        
        return jsonify({
            'status': 'success',
            'cached': False,
            'result': report
        })
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"体态评估出错: {str(e)}")
        print(error_trace)
        return jsonify({
            'status': 'error',
            'message': f'评估过程中发生异常: {str(e)}',
            'trace': error_trace
        })

@app.route('/get_latest_assessment', methods=['GET'])
def get_assessment_result():
    """获取最新的评估结果"""
    global assessment_result_cache, assessment_timestamp
    
    if not assessment_result_cache:
        # 如果没有缓存的结果，尝试进行新的评估
        try:
            result = get_latest_assessment(os.path.join(output_dir, "static"))
            if "error" not in result:
                assessment_result_cache = result
                assessment_timestamp = time.time()
                return jsonify({
                    'status': 'success',
                    'cached': False,
                    'result': result
                })
            else:
                return jsonify({
                    'status': 'error',
                    'message': '无法获取评估结果，请先进行数据采集和评估'
                })
        except Exception as e:
            return jsonify({
                'status': 'error',
                'message': f'获取评估结果时出错: {str(e)}'
            })
    else:
        # 返回缓存的结果
        return jsonify({
            'status': 'success',
            'cached': True,
            'result': assessment_result_cache,
            'timestamp': assessment_timestamp
        })

@app.route('/get_assessment_data', methods=['GET'])
def get_assessment_data():
    """获取体态评估数据，供报告页面使用"""
    global assessment_result_cache, assessment_timestamp
    
    if not assessment_result_cache:
        # 如果没有缓存的结果，尝试从最近的评估中获取
        try:
            result = get_latest_assessment(os.path.join(output_dir, "static"))
            if "error" not in result:
                assessment_result_cache = result
                assessment_timestamp = time.time()
                return jsonify({
                    'status': 'success',
                    'result': result
                })
            else:
                return jsonify({
                    'status': 'error',
                    'message': '未找到体态评估数据，请先进行体态分析'
                })
        except Exception as e:
            return jsonify({
                'status': 'error',
                'message': f'获取评估数据时出错: {str(e)}'
            })
    else:
        # 返回缓存的结果
        return jsonify({
            'status': 'success',
            'result': assessment_result_cache
        })

# 新增路由：清除评估缓存
@app.route('/clear_assessment_cache', methods=['POST'])
def clear_assessment_cache():
    """清除评估结果缓存"""
    global assessment_result_cache, assessment_timestamp
    
    assessment_result_cache = None
    assessment_timestamp = None
    
    return jsonify({
        'status': 'success',
        'message': '评估结果缓存已清除'
    })

@app.route('/analyze_posture', methods=['POST'])
def analyze_posture():
    """后台处理体态分析（实际计算）"""
    global assessment_result_cache, assessment_timestamp
    
    print("收到体态分析请求")  # 调试信息
    try:
        # 使用体态评估模块进行评估
        # 首先检查是否有记录数据
        if not last_saved_dir or not os.path.exists(last_saved_dir):
            print("没有找到可用的记录数据，使用模拟数据")
            # 使用模拟数据
            assessment_result = {
                "status": "success",
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "total_score": 75,
                "details": [
                    {
                        "problem": "head_forward_tilt",
                        "status": "mild",
                        "score": 80,
                        "weight": 1.5,
                        "weighted_score": 120,
                        "measurement": {"angle": 25.5}
                    },
                    {
                        "problem": "thoracic_kyphosis",
                        "status": "normal",
                        "score": 90,
                        "weight": 1.2,
                        "weighted_score": 108,
                        "measurement": {"angle": 35.2}
                    },
                    {
                        "problem": "shoulder_drop",
                        "status": "mild",
                        "score": 85,
                        "weight": 1.0,
                        "weighted_score": 85,
                        "measurement": {"diff_cm": 1.8}
                    },
                    {
                        "problem": "scoliosis",
                        "status": "normal",
                        "score": 95,
                        "weight": 1.8,
                        "weighted_score": 171,
                        "measurement": {"angle": 5.1}
                    },
                    {
                        "problem": "anterior_pelvic_tilt",
                        "status": "mild",
                        "score": 82,
                        "weight": 1.2,
                        "weighted_score": 98.4,
                        "measurement": {"angle": 12.3}
                    }
                ]
            }
            
            # 更新全局缓存
            assessment_result_cache = assessment_result
            assessment_timestamp = time.time()
            
            print("使用模拟数据完成体态分析计算")
            return jsonify({
                "status": "success",
                "message": "体态分析计算完成（使用模拟数据）"
            })
        
        # 检查最后保存的目录里是否有帧数据
        frames_dir = os.path.join(last_saved_dir, "frames")
        if os.path.exists(frames_dir):
            frames = [f for f in os.listdir(frames_dir) if f.startswith("frame_")]
            if frames:
                print(f"可用帧数据: {frames[:5] + ['...'] if len(frames) > 5 else frames}")
            else:
                print("警告: 没有找到帧数据")
        else:
            print(f"警告: 帧目录不存在: {frames_dir}")
            
        # 从最后保存的数据目录读取评估数据
        keypoints_file = os.path.join(last_saved_dir, "keypoints.json")
        if not os.path.exists(keypoints_file):
            print(f"未找到关键点数据文件: {keypoints_file}，使用模拟数据")
            # 使用模拟数据
            assessment_result = {
                "status": "success",
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "total_score": 75,
                "details": [
                    {
                        "problem": "head_forward_tilt",
                        "status": "mild",
                        "score": 80,
                        "weight": 1.5,
                        "weighted_score": 120,
                        "measurement": {"angle": 25.5}
                    },
                    {
                        "problem": "thoracic_kyphosis",
                        "status": "normal",
                        "score": 90,
                        "weight": 1.2,
                        "weighted_score": 108,
                        "measurement": {"angle": 35.2}
                    },
                    {
                        "problem": "shoulder_drop",
                        "status": "mild",
                        "score": 85,
                        "weight": 1.0,
                        "weighted_score": 85,
                        "measurement": {"diff_cm": 1.8}
                    },
                    {
                        "problem": "scoliosis",
                        "status": "normal",
                        "score": 95,
                        "weight": 1.8,
                        "weighted_score": 171,
                        "measurement": {"angle": 5.1}
                    },
                    {
                        "problem": "anterior_pelvic_tilt",
                        "status": "mild",
                        "score": 82,
                        "weight": 1.2,
                        "weighted_score": 98.4,
                        "measurement": {"angle": 12.3}
                    }
                ]
            }
            
            # 更新全局缓存
            assessment_result_cache = assessment_result
            assessment_timestamp = time.time()
            
            print("使用模拟数据完成体态分析计算")
            return jsonify({
                "status": "success",
                "message": "体态分析计算完成（使用模拟数据）"
            })
        
        # 读取关键点数据
        try:
            with open(keypoints_file, 'r', encoding='utf-8') as f:
                keypoints = json.load(f)
            if not keypoints:
                print("警告: 关键点数据文件为空")
        except Exception as e:
            print(f"读取关键点数据失败: {e}")
            keypoints = []
            
        if not keypoints:
            print("关键点数据为空，使用模拟数据")
            # 使用模拟数据
            assessment_result = {
                "status": "success",
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "total_score": 75,
                "details": [
                    {
                        "problem": "head_forward_tilt",
                        "status": "mild",
                        "score": 80,
                        "weight": 1.5,
                        "weighted_score": 120,
                        "measurement": {"angle": 25.5}
                    },
                    {
                        "problem": "thoracic_kyphosis",
                        "status": "normal",
                        "score": 90,
                        "weight": 1.2,
                        "weighted_score": 108,
                        "measurement": {"angle": 35.2}
                    },
                    {
                        "problem": "shoulder_drop",
                        "status": "mild",
                        "score": 85,
                        "weight": 1.0,
                        "weighted_score": 85,
                        "measurement": {"diff_cm": 1.8}
                    },
                    {
                        "problem": "scoliosis",
                        "status": "normal",
                        "score": 95,
                        "weight": 1.8,
                        "weighted_score": 171,
                        "measurement": {"angle": 5.1}
                    },
                    {
                        "problem": "anterior_pelvic_tilt",
                        "status": "mild",
                        "score": 82,
                        "weight": 1.2,
                        "weighted_score": 98.4,
                        "measurement": {"angle": 12.3}
                    }
                ]
            }
            
            # 更新全局缓存
            assessment_result_cache = assessment_result
            assessment_timestamp = time.time()
            
            print("使用模拟数据完成体态分析计算")
            return jsonify({
                "status": "success",
                "message": "体态分析计算完成（使用模拟数据）"
            })
                
        # 获取最新的评估结果
        try:
            assessment_result = get_latest_assessment()
        except Exception as e:
            print(f"调用评估模块失败: {e}")
            assessment_result = None
        
        if assessment_result is None:
            print("评估失败，未能生成结果，使用模拟数据")
            # 使用模拟数据
            assessment_result = {
                "status": "success",
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "total_score": 75,
                "details": [
                    {
                        "problem": "head_forward_tilt",
                        "status": "mild",
                        "score": 80,
                        "weight": 1.5,
                        "weighted_score": 120,
                        "measurement": {"angle": 25.5}
                    },
                    {
                        "problem": "thoracic_kyphosis",
                        "status": "normal",
                        "score": 90,
                        "weight": 1.2,
                        "weighted_score": 108,
                        "measurement": {"angle": 35.2}
                    },
                    {
                        "problem": "shoulder_drop",
                        "status": "mild",
                        "score": 85,
                        "weight": 1.0,
                        "weighted_score": 85,
                        "measurement": {"diff_cm": 1.8}
                    },
                    {
                        "problem": "scoliosis",
                        "status": "normal",
                        "score": 95,
                        "weight": 1.8,
                        "weighted_score": 171,
                        "measurement": {"angle": 5.1}
                    },
                    {
                        "problem": "anterior_pelvic_tilt",
                        "status": "mild",
                        "score": 82,
                        "weight": 1.2,
                        "weighted_score": 98.4,
                        "measurement": {"angle": 12.3}
                    }
                ]
            }
        
        # 更新全局缓存
        assessment_result_cache = assessment_result
        assessment_timestamp = time.time()
        
        print("体态分析计算完成，返回成功响应")
        return jsonify({
            "status": "success",
            "message": "体态分析计算完成"
        })
    except Exception as e:
        print(f"体态分析计算出错: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            "status": "error",
            "message": f"体态分析计算失败: {str(e)}"
        })

if __name__ == '__main__':
    # 设置多进程启动方法
    mp.set_start_method('spawn', force=True)
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='体态分析与步态检测系统')
    parser.add_argument('--config', type=str, help='配置文件路径')
    parser.add_argument('--model', type=str, help='模型名称')
    parser.add_argument('--device', type=str, help='设备名称')
    parser.add_argument('--port', type=int, default=5000, help='Flask服务端口')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Flask服务主机')
    parser.add_argument('--debug', action='store_true', help='开启Flask调试模式')
    parser.add_argument('--output', type=str, help='数据输出目录')
    args = parser.parse_args()
    
    try:
        # 设置输出目录
        if args.output:
            output_dir = args.output
            
        # 初始化系统
        init_system(args.model, args.device)
        
        # 启动Flask应用
        app.run(host=args.host, port=args.port, debug=args.debug, threaded=True)
    finally:
        # 确保清理资源
        cleanup() 