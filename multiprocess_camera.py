import time
import cv2
import torch
import numpy as np
from mmpose.apis.inferencers import MMPoseInferencer
from mmpose.apis.inference_tracking import _compute_iou
import config
from utils import torch_inference_mode, init_torch_settings
from pose_visualization import process_pose_results
import base64

def camera_process(camera_id, return_dict, shared_data, model_name=None, device='cuda:0', custom_frame_processor=None):
    """每个摄像头的独立处理进程
    
    Args:
        camera_id: 摄像头ID
        return_dict: 用于返回处理结果的字典
        shared_data: 进程间共享数据
        model_name: 模型名称
        device: 设备名称
        custom_frame_processor: 自定义帧处理函数，如果提供则用它处理帧
    """
    try:
        # 设置模型名称，如果未提供则使用默认值
        model_name = model_name or config.ModelConfig.DEFAULT_MODEL
        
        # 设置CUDNN加速
        init_torch_settings(device)
        
        # 初始化摄像头 - 使用DirectShow后端
        cap = cv2.VideoCapture(camera_id, cv2.CAP_DSHOW)
        
        # 检查摄像头是否成功打开
        if not cap.isOpened():
            print(f"无法打开摄像头 {camera_id}")
            return_dict[f'error_{camera_id}'] = f"无法打开摄像头 {camera_id}"
            return
            
        # 尝试设置更高的捕获分辨率和其他优化
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        # 设置缓冲区大小为1，减少延迟
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        # 设置合适的分辨率和帧率
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CameraConfig.DEFAULT_CAMERA_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CameraConfig.DEFAULT_CAMERA_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS, config.CameraConfig.DEFAULT_CAMERA_FPS)
        
        # 读取一帧验证摄像头设置是否生效
        ret, test_frame = cap.read()
        if ret:
            actual_width = test_frame.shape[1]  # 实际宽度
            actual_height = test_frame.shape[0]  # 实际高度
            
            print(f"摄像头 {camera_id} 实际分辨率: {actual_width}x{actual_height}, 期望分辨率: {config.CameraConfig.DEFAULT_CAMERA_WIDTH}x{config.CameraConfig.DEFAULT_CAMERA_HEIGHT}")
            
            # 如果摄像头不支持设置的分辨率，则后续会对每一帧进行统一尺寸的调整

        # 在独立进程中初始化模型
        try:
            # 为当前进程显式设置CUDA设备，避免内存碎片
            if 'cuda' in device:
                torch.cuda.set_device(int(device.split(':')[1]) if ':' in device else 0)
                
            # 加载模型时设置专用配置，提高初始化速度
            inferencer = MMPoseInferencer(
                pose2d=model_name,
                device=device,
                scope='mmpose',
                show_progress=False
            )
            print(f"进程 {camera_id} 成功加载模型到 {device}")
            
            # 创建用于模型推理的固定大小张量，避免动态分配内存
            try:
                # 尝试解析模型名称中的输入尺寸
                # 处理两种格式: 'name_384x288' 或 'name-384'
                if '_' in model_name and 'x' in model_name.split('_')[-1]:
                    # 格式如: rtmpose-l_8xb32-270e_coco-wholebody-384x288
                    width, height = [int(x) for x in model_name.split('_')[-1].split('x')]
                elif '-' in model_name and model_name.split('-')[-1].isdigit():
                    # 格式如: coco-wholebody-384
                    size = int(model_name.split('-')[-1])
                    width, height = size, size
                else:
                    # 使用默认的尺寸
                    width, height = config.CameraConfig.DEFAULT_CAMERA_WIDTH, config.CameraConfig.DEFAULT_CAMERA_HEIGHT
                    print(f"无法从模型名称 '{model_name}' 解析尺寸，使用默认尺寸 {width}x{height}")
            except Exception as e:
                # 解析失败时使用默认尺寸
                width, height = config.CameraConfig.DEFAULT_CAMERA_WIDTH, config.CameraConfig.DEFAULT_CAMERA_HEIGHT
                print(f"解析模型尺寸时出错: {str(e)}，使用默认尺寸 {width}x{height}")
                
            dummy_frame = np.zeros((height, width, 3), dtype=np.uint8)
            # dummy_frame = np.zeros((config.CameraConfig.DEFAULT_CAMERA_HEIGHT, 
            #                          config.CameraConfig.DEFAULT_CAMERA_WIDTH, 3), dtype=np.uint8)
            
            # 预热模型以初始化CUDA核心和缓存
            with torch_inference_mode():
                for _ in range(config.ModelConfig.MODEL_WARMUP_COUNT):  # 预热多次
                    _ = list(inferencer(dummy_frame))
                    
                # 强制同步GPU，确保预热完成    
                if 'cuda' in device:
                    torch.cuda.synchronize()
                
        except Exception as e:
            print(f"进程 {camera_id} 模型加载失败: {str(e)}")
            return_dict[f'error_{camera_id}'] = f"模型加载失败: {str(e)}"
            return
            
        # 推理配置 - 从全局配置复制
        call_args = config.InferenceConfig.DEFAULT_INFERENCE_CONFIG.copy()
        
        # 性能统计
        inference_times = []
        frame_count = 0
        start_time = time.time()
        
        # 创建图像转换缓存 - 预分配内存
        img_cache = np.empty((config.CameraConfig.DEFAULT_CAMERA_HEIGHT, 
                               config.CameraConfig.DEFAULT_CAMERA_WIDTH, 3), dtype=np.uint8)
        
        # 跟踪状态变量
        results_last = []  # 上一帧的结果
        next_id = 0        # 下一个可用的ID
        tracking_thr = config.TrackingConfig.TRACKING_THRESHOLD  # IOU阈值
        
        # 自定义跟踪函数
        def track_by_iou(bbox, results_last, thr):
            """使用IOU跟踪对象"""
            max_iou_score = -1
            max_index = -1
            track_id = -1
            
            # 确保bbox格式正确
            if not isinstance(bbox, (list, np.ndarray)) or len(bbox) < 4:
                return -1, results_last
            
            for index, res_last in enumerate(results_last):
                last_bbox = res_last.get('bbox', None)
                if last_bbox is None or not isinstance(last_bbox, (list, np.ndarray)) or len(last_bbox) < 4:
                    continue
                
                # 计算IOU，使用MMPose提供的函数
                try:
                    # 使用MMPose的IOU计算函数
                    iou_score = _compute_iou(bbox, last_bbox)
                    if iou_score > max_iou_score:
                        max_iou_score = iou_score
                        max_index = index
                except Exception as e:
                    print(f"计算IOU时出错: {str(e)}, bbox1: {bbox}, bbox2: {last_bbox}")
                    continue
            
            # 如果IOU得分大于阈值，使用匹配的上一帧对象的ID
            if max_iou_score > thr and max_index != -1:
                track_id = results_last[max_index].get('track_id', -1)
                # 从结果列表中移除已匹配的项
                results_last.pop(max_index)
            
            return track_id, results_last
        
        # 用于图像编码的变量
        encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), config.CameraConfig.JPEG_QUALITY]
        
        # 主处理循环
        while shared_data.running.value:
            # 读取帧
            ret, frame = cap.read()
            if not ret:
                print(f"摄像头 {camera_id} 无法读取帧，尝试重新初始化...")
                # 尝试重新初始化摄像头
                cap.release()
                time.sleep(config.CameraConfig.CAMERA_RECONNECT_DELAY)
                cap = cv2.VideoCapture(camera_id, cv2.CAP_DSHOW)
                if not cap.isOpened():
                    print(f"摄像头 {camera_id} 无法重新打开，退出进程")
                    break
                # 重新设置摄像头参数
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CameraConfig.DEFAULT_CAMERA_WIDTH)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CameraConfig.DEFAULT_CAMERA_HEIGHT)
                cap.set(cv2.CAP_PROP_FPS, config.CameraConfig.DEFAULT_CAMERA_FPS)
                continue
            
            # 确保所有帧的尺寸一致，如果与默认尺寸不符则调整
            if frame.shape[1] != config.CameraConfig.DEFAULT_CAMERA_WIDTH or frame.shape[0] != config.CameraConfig.DEFAULT_CAMERA_HEIGHT:
                frame = cv2.resize(frame, (config.CameraConfig.DEFAULT_CAMERA_WIDTH, config.CameraConfig.DEFAULT_CAMERA_HEIGHT))
                
            frame_count += 1
            
            # 处理帧
            start_inference = time.time()
            
            # 如果帧尺寸不符合预期，重新调整图像缓存
            if img_cache.shape[:2] != frame.shape[:2]:
                img_cache = np.empty(frame.shape, dtype=np.uint8)
            
            # 直接在预分配的内存上进行颜色转换
            cv2.cvtColor(frame, cv2.COLOR_BGR2RGB, img_cache)
            
            # 推理 - 使用优化的上下文管理器
            try:
                with torch_inference_mode():
                    raw_results = list(inferencer(img_cache, **call_args))
                    # 确保GPU操作完成，减少延迟抖动
                    if 'cuda' in device:
                        torch.cuda.synchronize()
            except Exception as e:
                print(f"推理过程中出错: {str(e)}")
                continue  # 跳过这一帧，继续处理下一帧
            
            inference_time = time.time() - start_inference
            inference_times.append(inference_time)
            if len(inference_times) > config.InferenceConfig.STATS_WINDOW_SIZE:
                inference_times.pop(0)
            
            # 添加跟踪ID处理
            current_results = []
            
            try:
                for result in raw_results:
                    pred_instances = result.get('predictions', [])
                    
                    # 遍历pred_instances中的所有实例
                    if pred_instances and len(pred_instances) > 0:
                        # MMPose 2.x返回的结果结构可能是一个列表，包含多个模型的预测结果
                        # 我们需要确保正确处理这种结构
                        for instance_list in pred_instances:
                            # 检查instance_list是否为列表类型
                            if isinstance(instance_list, list):
                                # 遍历列表中的每个实例（每个人）
                                for instance in instance_list:
                                    # 获取边界框
                                    bbox = instance.get('bbox', None)
                                    keypoints = instance.get('keypoints', None)
                                    
                                    # 如果没有边界框但有关键点，则使用关键点创建一个边界框
                                    if bbox is None and keypoints is not None and len(keypoints) > 0:
                                        try:
                                            keypoints = np.array(keypoints)
                                            if keypoints.size > 0:
                                                # 计算包含所有关键点的边界框
                                                valid_mask = np.isfinite(keypoints).all(axis=1)
                                                if np.any(valid_mask):  # 确保至少有一个有效的关键点
                                                    valid_keypoints = keypoints[valid_mask]
                                                    x_min = np.min(valid_keypoints[:, 0])
                                                    y_min = np.min(valid_keypoints[:, 1])
                                                    x_max = np.max(valid_keypoints[:, 0])
                                                    y_max = np.max(valid_keypoints[:, 1])
                                                    bbox = [x_min, y_min, x_max, y_max]
                                                    instance['bbox'] = bbox
                                        except Exception as e:
                                            print(f"从关键点创建边界框时出错: {str(e)}")
                                    
                                    if bbox is not None:
                                        try:
                                            # 确保边界框格式正确
                                            if isinstance(bbox, (list, np.ndarray)) and len(bbox) >= 4:
                                                # 尝试跟踪
                                                track_id, results_last = track_by_iou(bbox, results_last, tracking_thr)
                                                if track_id == -1:
                                                    # 如果没有匹配，分配新ID
                                                    track_id = next_id
                                                    next_id += 1
                                                
                                                # 设置跟踪ID
                                                instance['track_id'] = track_id
                                                
                                                # 保存当前实例以供下一帧使用
                                                current_results.append(instance)
                                        except Exception as e:
                                            print(f"处理跟踪ID时出错: {str(e)}, bbox: {bbox}")
                            elif isinstance(instance_list, dict):
                                # 如果直接是一个字典(单个人的情况)，直接处理
                                instance = instance_list
                                bbox = instance.get('bbox', None)
                                keypoints = instance.get('keypoints', None)
                                
                                # 如果没有边界框但有关键点，则使用关键点创建一个边界框
                                if bbox is None and keypoints is not None and len(keypoints) > 0:
                                    try:
                                        keypoints = np.array(keypoints)
                                        if keypoints.size > 0:
                                            # 计算包含所有关键点的边界框
                                            valid_mask = np.isfinite(keypoints).all(axis=1)
                                            if np.any(valid_mask):  # 确保至少有一个有效的关键点
                                                valid_keypoints = keypoints[valid_mask]
                                                x_min = np.min(valid_keypoints[:, 0])
                                                y_min = np.min(valid_keypoints[:, 1])
                                                x_max = np.max(valid_keypoints[:, 0])
                                                y_max = np.max(valid_keypoints[:, 1])
                                                bbox = [x_min, y_min, x_max, y_max]
                                                instance['bbox'] = bbox
                                    except Exception as e:
                                        print(f"从关键点创建边界框时出错: {str(e)}")
                                
                                if bbox is not None:
                                    try:
                                        # 确保边界框格式正确
                                        if isinstance(bbox, (list, np.ndarray)) and len(bbox) >= 4:
                                            # 尝试跟踪
                                            track_id, results_last = track_by_iou(bbox, results_last, tracking_thr)
                                            if track_id == -1:
                                                # 如果没有匹配，分配新ID
                                                track_id = next_id
                                                next_id += 1
                                            
                                            # 设置跟踪ID
                                            instance['track_id'] = track_id
                                            
                                            # 保存当前实例以供下一帧使用
                                            current_results.append(instance)
                                    except Exception as e:
                                        print(f"处理跟踪ID时出错: {str(e)}, bbox: {bbox}")
            except Exception as e:
                print(f"处理检测结果时出错: {str(e)}")
            
            # 更新跟踪状态
            results_last = current_results
                
            # 处理结果并渲染到帧上
            try:
                if custom_frame_processor:
                    # 如果提供了自定义处理函数，使用它
                    display_frame = custom_frame_processor(frame, raw_results, camera_id)
                else:
                    # 否则使用默认处理函数
                    display_frame = process_pose_results(frame, raw_results, call_args)
            except Exception as e:
                print(f"处理姿态结果时出错: {str(e)}")
                display_frame = frame.copy()  # 降级为原始帧
            
            # 计算FPS
            elapsed_time = time.time() - start_time
            fps = frame_count / elapsed_time if elapsed_time > 0 else 0
            
            # 每100帧重置计数器，防止数值过大
            if frame_count >= config.InferenceConfig.FPS_RESET_INTERVAL:
                start_time = time.time()
                frame_count = 0
                
            # 获取平均推理时间
            avg_inference_time = sum(inference_times) / len(inference_times) if inference_times else 0
            
            # 添加FPS和性能信息
            avg_inference_time_ms = avg_inference_time * 1000  # 转换为毫秒
            info_text = f"CAM {camera_id} | FPS: {fps:.1f} | Inference: {avg_inference_time_ms:.0f}ms"
            
            # 绘制半透明背景
            text_size = cv2.getTextSize(info_text, cv2.FONT_HERSHEY_SIMPLEX, 
                                      config.DisplayConfig.INFO_TEXT_SCALE, 
                                      config.DisplayConfig.INFO_TEXT_THICKNESS)[0]
            text_x, text_y = config.DisplayConfig.INFO_TEXT_POSITION
            pad_x, pad_y = config.DisplayConfig.INFO_BACKGROUND_PADDING
            
            # 创建背景矩形区域
            rect_x1 = text_x - pad_x
            rect_y1 = text_y - text_size[1] - pad_y
            rect_x2 = text_x + text_size[0] + pad_x
            rect_y2 = text_y + pad_y
            
            # 绘制半透明背景
            overlay = display_frame.copy()
            cv2.rectangle(overlay, (rect_x1, rect_y1), (rect_x2, rect_y2), 
                         config.DisplayConfig.INFO_BACKGROUND_COLOR[:3], -1)
            
            # 应用透明度
            alpha = config.DisplayConfig.INFO_BACKGROUND_COLOR[3]
            cv2.addWeighted(overlay, alpha, display_frame, 1 - alpha, 0, display_frame)
            
            # 绘制文本
            cv2.putText(display_frame, info_text, 
                        config.DisplayConfig.INFO_TEXT_POSITION, 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        config.DisplayConfig.INFO_TEXT_SCALE, 
                        config.DisplayConfig.INFO_TEXT_COLOR, 
                        config.DisplayConfig.INFO_TEXT_THICKNESS)
            
            # 将结果存储在共享字典中
            try:
                # 优化：使用更低的JPEG质量，进一步加快编码速度
                for retry in range(3):  # 尝试最多3次
                    try:
                        _, buffer = cv2.imencode('.jpg', display_frame, [cv2.IMWRITE_JPEG_QUALITY, config.CameraConfig.JPEG_QUALITY])
                        if buffer is None or buffer.size == 0:
                            print(f"摄像头 {camera_id} 图像编码失败 (尝试 {retry+1}/3)")
                            continue
                            
                        # 使用base64编码
                        base64_data = base64.b64encode(buffer).decode('utf-8')
                        
                        # 验证编码的数据
                        if not base64_data or len(base64_data) < 100:
                            print(f"摄像头 {camera_id} base64编码后数据异常短: {len(base64_data) if base64_data else 0} 字节")
                            continue
                            
                        # 存储结果
                        return_dict[f'frame_{camera_id}'] = base64_data
                        break  # 成功编码，跳出重试循环
                    except Exception as encode_err:
                        print(f"摄像头 {camera_id} 编码错误 (尝试 {retry+1}/3): {str(encode_err)}")
                        if retry == 2:  # 最后一次尝试
                            raise  # 重新抛出异常
                        time.sleep(0.01)  # 短暂延迟后重试
            except Exception as e:
                print(f"摄像头 {camera_id} 存储帧时出错: {str(e)}")
                # 如果帧编码/存储失败，尝试提供一个基本错误帧
                try:
                    error_img = np.zeros((config.CameraConfig.DEFAULT_CAMERA_HEIGHT, 
                                         config.CameraConfig.DEFAULT_CAMERA_WIDTH, 3), dtype=np.uint8)
                    cv2.putText(error_img, f"Camera {camera_id} Error", 
                                (50, config.CameraConfig.DEFAULT_CAMERA_HEIGHT//2), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    _, error_buffer = cv2.imencode('.jpg', error_img)
                    error_base64 = base64.b64encode(error_buffer).decode('utf-8')
                    return_dict[f'frame_{camera_id}'] = error_base64
                except:
                    print(f"摄像头 {camera_id} 无法提供错误帧")
            
            # 主动释放不需要的大型对象和清理内存
            del raw_results
            if 'cuda' in device:
                # 定期清理CUDA缓存
                if frame_count % config.TrackingConfig.CUDA_CACHE_CLEAR_INTERVAL == 0:
                    torch.cuda.empty_cache()
            
            # 为了调试输出检测到的人数
            if len(current_results) > 0 and config.SystemConfig.SHOW_DETECTION_COUNT:
                print(f"摄像头 {camera_id} 检测到 {len(current_results)} 人")
            
    except Exception as e:
        print(f"进程 {camera_id} 出错: {str(e)}")
        return_dict[f'error_{camera_id}'] = str(e)
        
    finally:
        # 释放资源
        if 'cap' in locals():
            cap.release()
        print(f"进程 {camera_id} 已退出") 