import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import config

class KeypointIndices:
    """关键点索引定义"""
    # 头部关键点
    NOSE = 0
    LEFT_EYE = 1
    RIGHT_EYE = 2
    LEFT_EAR = 3
    RIGHT_EAR = 4

    # 上半身关键点
    LEFT_SHOULDER = 5
    RIGHT_SHOULDER = 6
    LEFT_ELBOW = 7
    RIGHT_ELBOW = 8
    LEFT_WRIST = 9
    RIGHT_WRIST = 10

    # 下半身关键点
    LEFT_HIP = 11
    RIGHT_HIP = 12
    LEFT_KNEE = 13
    RIGHT_KNEE = 14
    LEFT_ANKLE = 15
    RIGHT_ANKLE = 16

    # 足部关键点
    LEFT_BIG_TOE = 17
    LEFT_SMALL_TOE = 18
    LEFT_HEEL = 19
    RIGHT_BIG_TOE = 20
    RIGHT_SMALL_TOE = 21
    RIGHT_HEEL = 22

    # 扩展关键点（自定义计算点）
    NECK_VERTEBRA = 23  # 颈椎中点
    SHOULDER_MID = 24   # 左右肩中点
    HIP_MID = 25        # 髋部中点

    # 常用索引组
    KEPT_INDICES = list(range(0, 23))  # 0-22的索引

class SkeletonDefinition:
    """骨架连接关系定义"""
    @staticmethod
    def get_standard_skeleton():
        """返回标准MMPose骨架连接"""
        K = KeypointIndices
        return [
            # 头部连接
            (K.NOSE, K.LEFT_EYE),           # 鼻子到左眼
            (K.NOSE, K.RIGHT_EYE),          # 鼻子到右眼
            (K.LEFT_EYE, K.LEFT_EAR),       # 左眼到左耳
            (K.RIGHT_EYE, K.RIGHT_EAR),     # 右眼到右耳
            
            # 上身躯干
            (K.LEFT_SHOULDER, K.RIGHT_SHOULDER),  # 左肩到右肩
            
            (K.LEFT_SHOULDER, K.LEFT_ELBOW),     # 左肩到左肘
            (K.LEFT_ELBOW, K.LEFT_WRIST),        # 左肘到左手腕
            
            (K.RIGHT_SHOULDER, K.RIGHT_ELBOW),   # 右肩到右肘
            (K.RIGHT_ELBOW, K.RIGHT_WRIST),      # 右肘到右手腕
            
            # 下半身
            (K.LEFT_SHOULDER, K.LEFT_HIP),       # 左肩到左髋
            (K.RIGHT_SHOULDER, K.RIGHT_HIP),     # 右肩到右髋
            (K.LEFT_HIP, K.RIGHT_HIP),           # 左髋到右髋
            
            (K.LEFT_HIP, K.LEFT_KNEE),           # 左髋到左膝
            (K.LEFT_KNEE, K.LEFT_ANKLE),         # 左膝到左踝
            
            (K.RIGHT_HIP, K.RIGHT_KNEE),         # 右髋到右膝
            (K.RIGHT_KNEE, K.RIGHT_ANKLE),       # 右膝到右踝
            
            # 足部连接
            (K.LEFT_ANKLE, K.LEFT_HEEL),         # 左踝到左脚跟
            (K.LEFT_ANKLE, K.LEFT_BIG_TOE),      # 左踝到左大拇指
            (K.LEFT_ANKLE, K.LEFT_SMALL_TOE),    # 左踝到左小指
            
            (K.RIGHT_ANKLE, K.RIGHT_HEEL),       # 右踝到右脚跟
            (K.RIGHT_ANKLE, K.RIGHT_BIG_TOE),    # 右踝到右大拇指
            (K.RIGHT_ANKLE, K.RIGHT_SMALL_TOE)   # 右踝到右小指
        ]
    
    @staticmethod
    def get_custom_skeleton():
        """返回自定义计算点骨架连接"""
        K = KeypointIndices
        return [
            (K.NECK_VERTEBRA, K.SHOULDER_MID),   # 颈椎中点到肩部中点
            (K.SHOULDER_MID, K.HIP_MID),         # 肩部中点到髋部中点
            (K.NECK_VERTEBRA, K.NOSE)            # 颈椎中点到鼻子
        ]


class PoseVisualizer:
    """姿态可视化类"""
    
    def __init__(self):
        """初始化姿态可视化器"""
        self.k = KeypointIndices
        self.standard_skeleton = SkeletonDefinition.get_standard_skeleton()
        self.custom_skeleton = SkeletonDefinition.get_custom_skeleton()
        self.link_colors = config.ColorConfig.SKELETON_COLORS
        self.custom_link_colors = [config.ColorConfig.SPINE_COLOR] * len(self.custom_skeleton)
    
    def process_pose_results(self, frame: np.ndarray, results: List, call_args: Dict) -> np.ndarray:
        """处理姿态估计结果并绘制到帧上
        
        Args:
            frame: 原始视频帧
            results: 姿态估计结果
            call_args: 调用参数
            
        Returns:
            处理后的帧
        """
        display_frame = frame.copy()
        person_count = 0

        try:
            for result in results:
                pred_instances = result.get('predictions', [])
                
                if not pred_instances:
                    continue
                    
                for instance_list in pred_instances:
                    # 检查是否为列表类型（多个人的情况）
                    if isinstance(instance_list, list):
                        for instance in instance_list:
                            self._process_single_person(
                                display_frame, instance, person_count, 
                                self.k.KEPT_INDICES, call_args
                            )
                            person_count += 1
                    # 检查是否为字典类型（单个人的情况）
                    elif isinstance(instance_list, dict):
                        self._process_single_person(
                            display_frame, instance_list, person_count,
                            self.k.KEPT_INDICES, call_args
                        )
                        person_count += 1
        except Exception as e:
            print(f"处理姿态结果绘制时出错: {str(e)}")
        
        return display_frame

    def _process_single_person(self, display_frame: np.ndarray, instance: Dict, 
                              person_idx: int, kept_indices: List[int], call_args: Dict) -> None:
        """处理单个人的姿态估计结果
        
        Args:
            display_frame: 显示帧
            instance: 单个人的姿态估计结果
            person_idx: 人员索引
            kept_indices: 保留的关键点索引
            call_args: 调用参数
        """
        try:
            # 获取关键点和得分
            keypoints = instance.get('keypoints')
            keypoint_scores = instance.get('keypoint_scores')
            
            if not self._validate_keypoints(keypoints, keypoint_scores, kept_indices):
                return
                
            # 转换为numpy数组
            keypoints = np.array(keypoints)
            keypoint_scores = np.array(keypoint_scores)
            
            # 创建扩展关键点数组
            extended_data = self._compute_extended_keypoints(
                keypoints, keypoint_scores, call_args['kpt_thr'], display_frame, call_args['radius']
            )
            
            if extended_data is None:
                return
                
            extended_keypoints, extended_scores = extended_data
            
            # 绘制原始关键点
            self._draw_keypoints(display_frame, keypoints, keypoint_scores, 
                                kept_indices, call_args['kpt_thr'], call_args['radius'])
            
            # 绘制原始骨架连线
            self._draw_skeleton(display_frame, keypoints, keypoint_scores, 
                               self.standard_skeleton, self.link_colors, 
                               call_args['kpt_thr'], call_args['thickness'])
            
            # 绘制自定义骨架连线
            self._draw_extended_skeleton(display_frame, extended_keypoints, extended_scores,
                                        self.custom_skeleton, self.custom_link_colors, 
                                        call_args['thickness'])
            
            # 绘制边界框（如果需要）
            if call_args['draw_bbox']:
                self._draw_bbox(display_frame, instance.get('bbox'))
                
        except Exception as e:
            print(f"绘制单个人姿态时出错: {str(e)}")
    
    def _validate_keypoints(self, keypoints: Union[List, np.ndarray], 
                           keypoint_scores: Union[List, np.ndarray], 
                           kept_indices: List[int]) -> bool:
        """验证关键点数据是否有效
        
        Args:
            keypoints: 关键点坐标
            keypoint_scores: 关键点置信度分数
            kept_indices: 需要保留的关键点索引
            
        Returns:
            数据是否有效
        """
        if keypoints is None or keypoint_scores is None:
            return False
            
        if len(keypoints) == 0 or len(keypoint_scores) == 0:
            return False
            
        return True
    
    def _compute_extended_keypoints(self, keypoints: np.ndarray, keypoint_scores: np.ndarray, 
                                   kpt_thr: float, display_frame: np.ndarray, radius: int) -> Optional[Tuple]:
        """计算扩展关键点
        
        Args:
            keypoints: 原始关键点
            keypoint_scores: 关键点置信度
            kpt_thr: 关键点置信度阈值
            display_frame: 显示帧
            radius: 关键点半径
            
        Returns:
            扩展的关键点和置信度
        """
        if len(keypoints) < max(self.k.KEPT_INDICES) + 1:
            return None
            
        # 创建扩展关键点和分数数组
        max_idx = max(self.k.NECK_VERTEBRA, self.k.SHOULDER_MID, self.k.HIP_MID) + 1
        extended_keypoints = np.zeros((max_idx, 2), dtype=np.float32)
        extended_scores = np.zeros(max_idx, dtype=np.float32)
        
        # 复制原始关键点和分数
        for idx in self.k.KEPT_INDICES:
            if idx < len(keypoints):
                extended_keypoints[idx] = keypoints[idx]
                if idx < len(keypoint_scores):
                    extended_scores[idx] = keypoint_scores[idx]
        
        # 计算肩部中点
        shoulder_mid_valid = (
            keypoint_scores[self.k.LEFT_SHOULDER] > kpt_thr and 
            keypoint_scores[self.k.RIGHT_SHOULDER] > kpt_thr
        )
        
        if shoulder_mid_valid:
            # 计算左右肩中点
            shoulder_mid_x = (keypoints[self.k.LEFT_SHOULDER][0] + keypoints[self.k.RIGHT_SHOULDER][0]) / 2
            shoulder_mid_y = (keypoints[self.k.LEFT_SHOULDER][1] + keypoints[self.k.RIGHT_SHOULDER][1]) / 2
            
            # 存储肩部中点
            extended_keypoints[self.k.SHOULDER_MID] = [shoulder_mid_x, shoulder_mid_y]
            extended_scores[self.k.SHOULDER_MID] = 1.0  # 设置为满分
            
            # 绘制肩部中点
            cv2.circle(display_frame, (int(shoulder_mid_x), int(shoulder_mid_y)), 
                      radius, config.ColorConfig.NECK_VERTEBRA_COLOR, -1)
        
        # 计算颈椎中点
        neck_vertebra_valid = (
            keypoint_scores[self.k.NOSE] > kpt_thr and 
            keypoint_scores[self.k.LEFT_SHOULDER] > kpt_thr and 
            keypoint_scores[self.k.RIGHT_SHOULDER] > kpt_thr
        )
        
        if neck_vertebra_valid and shoulder_mid_valid:
            # 计算颈椎中点
            neck_vertebra_x = keypoints[self.k.NOSE][0] * 0.3 + extended_keypoints[self.k.SHOULDER_MID][0] * 0.7
            neck_vertebra_y = keypoints[self.k.NOSE][1] * 0.3 + extended_keypoints[self.k.SHOULDER_MID][1] * 0.7
            
            # 存储颈椎中点
            extended_keypoints[self.k.NECK_VERTEBRA] = [neck_vertebra_x, neck_vertebra_y]
            extended_scores[self.k.NECK_VERTEBRA] = 1.0  # 设置为满分
            
            # 绘制颈椎中点
            cv2.circle(display_frame, (int(neck_vertebra_x), int(neck_vertebra_y)), 
                      radius, config.ColorConfig.NECK_VERTEBRA_COLOR, -1)
        
        # 计算髋部中点
        hip_valid = (
            keypoint_scores[self.k.LEFT_HIP] > kpt_thr and 
            keypoint_scores[self.k.RIGHT_HIP] > kpt_thr
        )
        
        if hip_valid:
            hip_mid_x = (keypoints[self.k.LEFT_HIP][0] + keypoints[self.k.RIGHT_HIP][0]) / 2
            hip_mid_y = (keypoints[self.k.LEFT_HIP][1] + keypoints[self.k.RIGHT_HIP][1]) / 2
            
            # 存储髋部中点
            extended_keypoints[self.k.HIP_MID] = [hip_mid_x, hip_mid_y]
            extended_scores[self.k.HIP_MID] = 1.0  # 设置为满分
            
            # 绘制髂前上棘连线中点
            cv2.circle(display_frame, (int(hip_mid_x), int(hip_mid_y)), 
                      radius, config.ColorConfig.HIP_MID_COLOR, -1)
                      
        return extended_keypoints, extended_scores
    
    def _draw_keypoints(self, display_frame: np.ndarray, keypoints: np.ndarray, 
                       keypoint_scores: np.ndarray, kept_indices: List[int], 
                       kpt_thr: float, radius: int) -> None:
        """绘制关键点
        
        Args:
            display_frame: 显示帧
            keypoints: 关键点坐标
            keypoint_scores: 关键点置信度
            kept_indices: 要绘制的关键点索引
            kpt_thr: 关键点置信度阈值
            radius: 关键点半径
        """
        for idx in kept_indices:
            if idx < len(keypoints) and keypoint_scores[idx] > kpt_thr:
                x, y = int(keypoints[idx][0]), int(keypoints[idx][1])
                cv2.circle(display_frame, (x, y), radius, config.ColorConfig.KEYPOINT_COLOR, -1)
    
    def _draw_skeleton(self, display_frame: np.ndarray, keypoints: np.ndarray, 
                      keypoint_scores: np.ndarray, skeleton: List[Tuple[int, int]], 
                      link_colors: List, kpt_thr: float, thickness: int) -> None:
        """绘制骨架连线
        
        Args:
            display_frame: 显示帧
            keypoints: 关键点坐标
            keypoint_scores: 关键点置信度
            skeleton: 骨架连接关系
            link_colors: 连线颜色
            kpt_thr: 关键点置信度阈值
            thickness: 线条粗细
        """
        for sk_idx, (start_idx, end_idx) in enumerate(skeleton):
            if (start_idx < len(keypoints) and end_idx < len(keypoints) and
                keypoint_scores[start_idx] > kpt_thr and 
                keypoint_scores[end_idx] > kpt_thr):
                
                start_pt = (int(keypoints[start_idx][0]), int(keypoints[start_idx][1]))
                end_pt = (int(keypoints[end_idx][0]), int(keypoints[end_idx][1]))
                
                color = link_colors[sk_idx % len(link_colors)]
                cv2.line(display_frame, start_pt, end_pt, color, thickness)
    
    def _draw_extended_skeleton(self, display_frame: np.ndarray, extended_keypoints: np.ndarray, 
                               extended_scores: np.ndarray, custom_skeleton: List[Tuple[int, int]], 
                               custom_link_colors: List, thickness: int) -> None:
        """绘制扩展骨架连线
        
        Args:
            display_frame: 显示帧
            extended_keypoints: 扩展关键点坐标
            extended_scores: 扩展关键点置信度
            custom_skeleton: 自定义骨架连接关系
            custom_link_colors: 自定义连线颜色
            thickness: 线条粗细
        """
        for csk_idx, (start_idx, end_idx) in enumerate(custom_skeleton):
            if extended_scores[start_idx] > 0 and extended_scores[end_idx] > 0:
                start_pt = (int(extended_keypoints[start_idx][0]), int(extended_keypoints[start_idx][1]))
                end_pt = (int(extended_keypoints[end_idx][0]), int(extended_keypoints[end_idx][1]))
                
                color = custom_link_colors[csk_idx % len(custom_link_colors)]
                cv2.line(display_frame, start_pt, end_pt, color, thickness)
    
    def _draw_bbox(self, display_frame: np.ndarray, bbox: Optional[Union[List, np.ndarray]]) -> None:
        """绘制边界框
        
        Args:
            display_frame: 显示帧
            bbox: 边界框坐标 [x1, y1, x2, y2] 或 [x1, y1, x2, y2, score]
        """
        if bbox is None:
            return
            
        try:
            if isinstance(bbox, (list, np.ndarray)):
                if len(bbox) >= 4:
                    # 提取坐标并确保它们是整数
                    x1, y1, x2, y2 = [
                        int(float(coord)) if isinstance(coord, (int, float, str)) 
                        else int(coord[0]) for coord in bbox[:4]
                    ]
                    cv2.rectangle(
                        display_frame, (x1, y1), (x2, y2), 
                        config.ColorConfig.BBOX_COLOR, config.ColorConfig.BBOX_THICKNESS
                    )
        except Exception as e:
            print(f"处理边界框时出错: {str(e)}, 边界框数据: {bbox}")


def process_pose_results(frame: np.ndarray, results: List, call_args: Dict) -> np.ndarray:
    """处理姿态估计结果并绘制到帧上的便捷函数
    
    Args:
        frame: 原始视频帧
        results: 姿态估计结果
        call_args: 调用参数
        
    Returns:
        处理后的帧
    """
    visualizer = PoseVisualizer()
    return visualizer.process_pose_results(frame, results, call_args) 