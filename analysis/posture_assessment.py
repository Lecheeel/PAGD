import os
import json
import numpy as np
import math
from datetime import datetime

# 体态评估权重表
POSTURE_WEIGHTS = {
    # 前面观
    "head_forward_tilt": 3,  # 头向前倾
    "thoracic_kyphosis": 2,  # 胸脊柱后凸
    "flat_back": 1,  # 平背
    "posterior_pelvic_tilt": 1,  # 骨盆后倾
    "anterior_pelvic_tilt": 2,  # 骨盆前倾
    "knee_hyperextension": 1,  # 膝过伸
    "lateral_leaning": 2,  # 身体左右倾斜
    
    # 后面观
    "shoulder_drop": 1,  # 肩下垂
    "shoulder_internal_rotation": 1,  # 肩内旋
    "shoulder_external_rotation": 1,  # 肩外旋
    "scoliosis": 3,  # 脊柱侧弯
    "lateral_pelvic_tilt": 2,  # 骨盆向侧方倾斜
    "pelvic_rotation": 1,  # 骨盆旋转
    "foot_arch_abnormality": 1,  # 足弓异常
    
    # 侧面观
    "jaw_asymmetry": 1,  # 头下颌骨不对称
    "clavicle_asymmetry": 1,  # 锁骨和其他关节不对称
    "hip_external_rotation": 2,  # 髋外旋
    "hip_internal_rotation": 2,  # 髋内旋
    "knee_valgus": 2,  # 膝外翻
    "knee_varus": 2,  # 膝内翻
    "tibial_external_rotation": 2,  # 胫骨外旋
    "tibial_internal_rotation": 3,  # 胫骨内旋
}

# 体态问题评分标准
POSTURE_STANDARDS = {
    "head_forward_tilt": {
        "normal": {"range": [30, 40], "score": 1},
        "mild": {"range": [20, 30], "score": 2},
        "moderate": {"range": [10, 20], "score": 3},
        "severe": {"range": [0, 10], "score": 4}
    },
    # 其他体态问题的评分标准可以根据权得分配表添加
    # ...
}

# 总权重占比
POSTURE_TOTAL_WEIGHT_PERCENT = {
    "front_view": 0.10,  # 前面观占10%
    "back_view": 0.08,   # 后面观占8%
    "side_view": 0.14    # 侧面观占14%
}

class PostureAssessment:
    def __init__(self):
        self.results = {}
        self.scores = {}
        self.total_score = 0
        self.keypoints_data = None
        
    def load_latest_data(self, data_dir="data/recorded/static"):
        """加载最新的静态姿态数据"""
        try:
            # 获取所有子目录（按时间戳排序）
            subdirs = [os.path.join(data_dir, d) for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
            if not subdirs:
                print(f"未找到数据：{data_dir}")
                return False
                
            # 按时间戳排序（最新的在最后）
            subdirs.sort()
            latest_dir = subdirs[-1]
            
            # 加载关键点数据
            keypoints_file = os.path.join(latest_dir, "keypoints.json")
            if not os.path.exists(keypoints_file):
                print(f"未找到关键点数据文件：{keypoints_file}")
                return False
                
            with open(keypoints_file, 'r', encoding='utf-8') as f:
                self.keypoints_data = json.load(f)
                
            print(f"成功加载最新数据，时间戳：{os.path.basename(latest_dir)}")
            print(f"包含 {len(self.keypoints_data)} 条关键点数据")
            return True
        except Exception as e:
            print(f"加载数据时出错：{str(e)}")
            return False
    
    def _calculate_angle(self, p1, p2, p3):
        """计算三点之间的角度（单位：度）"""
        # 将输入点转换为numpy数组以便计算
        p1 = np.array(p1)
        p2 = np.array(p2)
        p3 = np.array(p3)
        
        # 计算两个向量
        v1 = p1 - p2
        v2 = p3 - p2
        
        # 计算向量点积
        dot_product = np.dot(v1, v2)
        
        # 计算向量模长
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        
        # 计算夹角（弧度）
        angle_rad = np.arccos(dot_product / (norm_v1 * norm_v2))
        
        # 将弧度转换为角度
        angle_deg = np.degrees(angle_rad)
        
        return angle_deg
    
    def _get_keypoint_by_name(self, keypoints, name):
        """从关键点列表中获取指定名称的关键点"""
        # 这里需要根据实际的关键点数据结构进行调整
        for keypoint in keypoints:
            if keypoint["name"] == name:
                return [keypoint["x"], keypoint["y"], keypoint["z"]]
        return None
    
    def _average_keypoints(self):
        """计算多帧关键点数据的平均值，提高稳定性"""
        if not self.keypoints_data or len(self.keypoints_data) == 0:
            return None
            
        # 假设每条记录中的keypoints是一个关键点列表
        # 这里需要根据实际数据结构进行调整
        averaged_keypoints = {}
        keypoint_count = {}
        
        # 计算所有帧中每个关键点的总和
        for record in self.keypoints_data:
            keypoints = record["keypoints"]
            for keypoint in keypoints:
                name = keypoint["name"]
                if name not in averaged_keypoints:
                    averaged_keypoints[name] = [0, 0, 0]
                    keypoint_count[name] = 0
                
                averaged_keypoints[name][0] += keypoint["x"]
                averaged_keypoints[name][1] += keypoint["y"]
                averaged_keypoints[name][2] += keypoint["z"]
                keypoint_count[name] += 1
        
        # 计算平均值
        for name in averaged_keypoints:
            if keypoint_count[name] > 0:
                averaged_keypoints[name][0] /= keypoint_count[name]
                averaged_keypoints[name][1] /= keypoint_count[name]
                averaged_keypoints[name][2] /= keypoint_count[name]
        
        return averaged_keypoints
    
    def _assess_head_forward_tilt(self, keypoints):
        """评估头向前倾的程度"""
        try:
            # 获取相关关键点
            ear = self._get_keypoint_by_name(keypoints, "right_ear")
            shoulder = self._get_keypoint_by_name(keypoints, "right_shoulder")
            hip = self._get_keypoint_by_name(keypoints, "right_hip")
            
            if not all([ear, shoulder, hip]):
                return {"status": "missing_data", "score": 0, "angle": None}
            
            # 计算颈部角度（耳朵-肩膀-臀部）
            neck_angle = self._calculate_angle(ear, shoulder, hip)
            
            # 根据角度确定严重程度
            if 30 <= neck_angle <= 40:
                status = "normal"
                score = 1
            elif 20 <= neck_angle < 30:
                status = "mild"
                score = 2
            elif 10 <= neck_angle < 20:
                status = "moderate"
                score = 3
            else:
                status = "severe"
                score = 4
            
            return {"status": status, "score": score, "angle": neck_angle}
        except Exception as e:
            print(f"评估头向前倾时出错：{str(e)}")
            return {"status": "error", "score": 0, "angle": None}
    
    def _assess_thoracic_kyphosis(self, keypoints):
        """评估胸脊柱后凸的程度"""
        try:
            # 获取相关关键点
            neck = self._get_keypoint_by_name(keypoints, "neck")
            mid_spine = self._get_keypoint_by_name(keypoints, "mid_spine")
            pelvis = self._get_keypoint_by_name(keypoints, "pelvis")
            
            if not all([neck, mid_spine, pelvis]):
                return {"status": "missing_data", "score": 0, "angle": None}
            
            # 计算胸椎角度
            thoracic_angle = self._calculate_angle(neck, mid_spine, pelvis)
            
            # 根据角度确定严重程度
            if 20 <= thoracic_angle <= 40:
                status = "normal"
                score = 1
            elif 40 < thoracic_angle <= 50:
                status = "mild"
                score = 2
            elif 50 < thoracic_angle <= 70:
                status = "moderate"
                score = 3
            else:
                status = "severe"
                score = 4
            
            return {"status": status, "score": score, "angle": thoracic_angle}
        except Exception as e:
            print(f"评估胸脊柱后凸时出错：{str(e)}")
            return {"status": "error", "score": 0, "angle": None}
    
    def _assess_anterior_pelvic_tilt(self, keypoints):
        """评估骨盆前倾的程度"""
        try:
            # 获取相关关键点
            hip = self._get_keypoint_by_name(keypoints, "right_hip")
            knee = self._get_keypoint_by_name(keypoints, "right_knee")
            shoulder = self._get_keypoint_by_name(keypoints, "right_shoulder")
            
            if not all([hip, knee, shoulder]):
                return {"status": "missing_data", "score": 0, "angle": None}
            
            # 计算骨盆倾斜角度（肩膀-臀部-膝盖）
            pelvic_angle = self._calculate_angle(shoulder, hip, knee)
            
            # 根据角度确定严重程度
            if 170 <= pelvic_angle <= 180:  # 正常情况下应该接近直线
                status = "normal"
                score = 1
            elif 160 <= pelvic_angle < 170:  # 轻度前倾
                status = "mild"
                score = 2
            elif 150 <= pelvic_angle < 160:  # 中度前倾
                status = "moderate"
                score = 3
            else:  # 重度前倾
                status = "severe"
                score = 4
            
            return {"status": status, "score": score, "angle": pelvic_angle}
        except Exception as e:
            print(f"评估骨盆前倾时出错：{str(e)}")
            return {"status": "error", "score": 0, "angle": None}

    def _assess_posterior_pelvic_tilt(self, keypoints):
        """评估骨盆后倾的程度"""
        try:
            # 获取相关关键点
            hip = self._get_keypoint_by_name(keypoints, "right_hip")
            knee = self._get_keypoint_by_name(keypoints, "right_knee")
            shoulder = self._get_keypoint_by_name(keypoints, "right_shoulder")
            
            if not all([hip, knee, shoulder]):
                return {"status": "missing_data", "score": 0, "angle": None}
            
            # 计算骨盆倾斜角度（肩膀-臀部-膝盖）
            pelvic_angle = self._calculate_angle(shoulder, hip, knee)
            
            # 根据角度确定严重程度（骨盆后倾与前倾相反）
            if 170 <= pelvic_angle <= 180:  # 正常情况下应该接近直线
                status = "normal"
                score = 1
            elif 180 < pelvic_angle <= 190:  # 轻度后倾
                status = "mild"
                score = 2
            elif 190 < pelvic_angle <= 200:  # 中度后倾
                status = "moderate"
                score = 3
            else:  # 重度后倾
                status = "severe"
                score = 4
            
            return {"status": status, "score": score, "angle": pelvic_angle}
        except Exception as e:
            print(f"评估骨盆后倾时出错：{str(e)}")
            return {"status": "error", "score": 0, "angle": None}

    def _assess_lateral_pelvic_tilt(self, keypoints):
        """评估骨盆侧倾的程度"""
        try:
            # 获取相关关键点
            left_hip = self._get_keypoint_by_name(keypoints, "left_hip")
            right_hip = self._get_keypoint_by_name(keypoints, "right_hip")
            mid_hip = self._get_keypoint_by_name(keypoints, "mid_hip")  # 可能需要自行计算
            neck = self._get_keypoint_by_name(keypoints, "neck")
            
            if not all([left_hip, right_hip, neck]):
                return {"status": "missing_data", "score": 0, "tilt_cm": None}
            
            # 如果没有mid_hip关键点，则计算左右臀部的中点
            if not mid_hip:
                mid_hip = [
                    (left_hip[0] + right_hip[0]) / 2,
                    (left_hip[1] + right_hip[1]) / 2,
                    (left_hip[2] + right_hip[2]) / 2
                ]
            
            # 计算左右臀部的高度差异（Y轴值的差异）
            height_diff = abs(left_hip[1] - right_hip[1])
            
            # 将差异转换为厘米（假设1单位=1厘米，根据实际情况调整）
            tilt_cm = height_diff
            
            # 根据高度差异确定严重程度
            if tilt_cm <= 1:
                status = "normal"
                score = 1
            elif 1 < tilt_cm <= 2:
                status = "mild"
                score = 2
            elif 2 < tilt_cm <= 4:
                status = "moderate"
                score = 3
            else:
                status = "severe"
                score = 4
            
            return {"status": status, "score": score, "tilt_cm": tilt_cm}
        except Exception as e:
            print(f"评估骨盆侧倾时出错：{str(e)}")
            return {"status": "error", "score": 0, "tilt_cm": None}

    def _assess_scoliosis(self, keypoints):
        """评估脊柱侧弯的程度"""
        try:
            # 获取相关关键点
            neck = self._get_keypoint_by_name(keypoints, "neck")
            mid_spine = self._get_keypoint_by_name(keypoints, "mid_spine")
            mid_hip = self._get_keypoint_by_name(keypoints, "mid_hip")
            
            if not all([neck, mid_spine, mid_hip]):
                return {"status": "missing_data", "score": 0, "angle": None}
            
            # 计算脊柱的直线度
            # 这里我们计算脊柱的Cobb角（近似值）
            # 在真实情况下，Cobb角需要通过X光片和特定方法计算
            
            # 计算脊柱的偏离角度
            # 我们使用颈部到臀部的连线与竖直方向的夹角作为近似
            spine_vector = [mid_hip[0] - neck[0], mid_hip[1] - neck[1], mid_hip[2] - neck[2]]
            vertical_vector = [0, 1, 0]  # Y轴向下的单位向量
            
            # 计算向量与垂直方向的夹角
            dot_product = spine_vector[0] * vertical_vector[0] + spine_vector[1] * vertical_vector[1] + spine_vector[2] * vertical_vector[2]
            spine_magnitude = math.sqrt(spine_vector[0]**2 + spine_vector[1]**2 + spine_vector[2]**2)
            vertical_magnitude = math.sqrt(vertical_vector[0]**2 + vertical_vector[1]**2 + vertical_vector[2]**2)
            
            cosine_angle = dot_product / (spine_magnitude * vertical_magnitude)
            cosine_angle = min(1.0, max(-1.0, cosine_angle))  # 确保值在[-1, 1]范围内
            angle_rad = math.acos(cosine_angle)
            angle_deg = math.degrees(angle_rad)
            
            # 根据角度确定严重程度
            if angle_deg < 10:
                status = "normal"
                score = 1
            elif 10 <= angle_deg < 20:
                status = "mild"
                score = 2
            elif 20 <= angle_deg < 40:
                status = "moderate"
                score = 3
            else:
                status = "severe"
                score = 4
            
            return {"status": status, "score": score, "angle": angle_deg}
        except Exception as e:
            print(f"评估脊柱侧弯时出错：{str(e)}")
            return {"status": "error", "score": 0, "angle": None}

    def _assess_shoulder_drop(self, keypoints):
        """评估肩膀下垂的程度"""
        try:
            # 获取相关关键点
            left_shoulder = self._get_keypoint_by_name(keypoints, "left_shoulder")
            right_shoulder = self._get_keypoint_by_name(keypoints, "right_shoulder")
            
            if not all([left_shoulder, right_shoulder]):
                return {"status": "missing_data", "score": 0, "diff_cm": None}
            
            # 计算左右肩高度差异（Y轴值的差异）
            height_diff = abs(left_shoulder[1] - right_shoulder[1])
            
            # 将差异转换为厘米（假设1单位=1厘米，根据实际情况调整）
            diff_cm = height_diff
            
            # 根据高度差异确定严重程度
            if diff_cm <= 1:
                status = "normal"
                score = 1
            elif 1 < diff_cm <= 2:
                status = "mild"
                score = 2
            elif 2 < diff_cm <= 4:
                status = "moderate"
                score = 3
            else:
                status = "severe"
                score = 4
            
            return {"status": status, "score": score, "diff_cm": diff_cm}
        except Exception as e:
            print(f"评估肩膀下垂时出错：{str(e)}")
            return {"status": "error", "score": 0, "diff_cm": None}

    def _assess_knee_valgus(self, keypoints):
        """评估膝外翻（X型腿）的程度"""
        try:
            # 获取相关关键点
            left_hip = self._get_keypoint_by_name(keypoints, "left_hip")
            left_knee = self._get_keypoint_by_name(keypoints, "left_knee")
            left_ankle = self._get_keypoint_by_name(keypoints, "left_ankle")
            right_hip = self._get_keypoint_by_name(keypoints, "right_hip")
            right_knee = self._get_keypoint_by_name(keypoints, "right_knee")
            right_ankle = self._get_keypoint_by_name(keypoints, "right_ankle")
            
            if not all([left_hip, left_knee, left_ankle, right_hip, right_knee, right_ankle]):
                return {"status": "missing_data", "score": 0, "gap_mm": None}
            
            # 计算膝关节间隙（两膝盖的距离）
            knee_gap = math.sqrt((left_knee[0] - right_knee[0])**2 + (left_knee[1] - right_knee[1])**2 + (left_knee[2] - right_knee[2])**2)
            
            # 计算踝关节间隙（两踝的距离）
            ankle_gap = math.sqrt((left_ankle[0] - right_ankle[0])**2 + (left_ankle[1] - right_ankle[1])**2 + (left_ankle[2] - right_ankle[2])**2)
            
            # 计算膝外翻的程度（膝关节间隙与踝关节间隙的比值）
            # 正常情况下，膝关节间隙应该小于或等于踝关节间隙
            valgus_ratio = knee_gap / ankle_gap if ankle_gap > 0 else 0
            
            # 将比值转换为毫米间隙估计
            gap_mm = max(0, (valgus_ratio - 1) * 10)  # 假设比值每0.1对应1mm间隙
            
            # 根据间隙确定严重程度
            if gap_mm <= 5:
                status = "normal"
                score = 1
            elif 5 < gap_mm <= 10:
                status = "mild"
                score = 2
            elif 10 < gap_mm <= 15:
                status = "moderate"
                score = 3
            else:
                status = "severe"
                score = 4
            
            return {"status": status, "score": score, "gap_mm": gap_mm}
        except Exception as e:
            print(f"评估膝外翻时出错：{str(e)}")
            return {"status": "error", "score": 0, "gap_mm": None}
    
    def _assess_lateral_leaning(self, keypoints):
        """评估身体左右倾斜的程度（基于髋部中点与两脚构成的三角形是否为等腰三角形）"""
        try:
            # 获取相关关键点
            left_hip = self._get_keypoint_by_name(keypoints, "left_hip")
            right_hip = self._get_keypoint_by_name(keypoints, "right_hip")
            left_foot = self._get_keypoint_by_name(keypoints, "left_ankle")  # 使用脚踝作为脚的位置
            right_foot = self._get_keypoint_by_name(keypoints, "right_ankle")
            
            if not all([left_hip, right_hip, left_foot, right_foot]):
                return {"status": "missing_data", "score": 0, "difference_ratio": None}
            
            # 计算髋部中点
            mid_hip = [
                (left_hip[0] + right_hip[0]) / 2,
                (left_hip[1] + right_hip[1]) / 2,
                (left_hip[2] + right_hip[2]) / 2
            ]
            
            # 计算髋部中点到左脚和右脚的距离
            left_distance = math.sqrt(
                (mid_hip[0] - left_foot[0])**2 + 
                (mid_hip[1] - left_foot[1])**2 + 
                (mid_hip[2] - left_foot[2])**2
            )
            
            right_distance = math.sqrt(
                (mid_hip[0] - right_foot[0])**2 + 
                (mid_hip[1] - right_foot[1])**2 + 
                (mid_hip[2] - right_foot[2])**2
            )
            
            # 计算两边长度的比例差异（等腰三角形两边应相等）
            if max(left_distance, right_distance) > 0:
                difference_ratio = abs(left_distance - right_distance) / max(left_distance, right_distance)
            else:
                difference_ratio = 0
            
            # 确定倾斜方向
            lean_direction = "balanced"
            if left_distance < right_distance:
                lean_direction = "left"  # 向左倾斜
            elif left_distance > right_distance:
                lean_direction = "right"  # 向右倾斜
            
            # 根据差异比例确定严重程度
            if difference_ratio <= 0.05:  # 允许5%的误差范围
                status = "normal"
                score = 1
                description = "身体左右平衡良好"
            elif difference_ratio <= 0.10:
                status = "mild"
                score = 2
                description = f"轻度向{'左' if lean_direction == 'left' else '右'}侧倾斜"
            elif difference_ratio <= 0.20:
                status = "moderate"
                score = 3
                description = f"中度向{'左' if lean_direction == 'left' else '右'}侧倾斜"
            else:
                status = "severe"
                score = 4
                description = f"严重向{'左' if lean_direction == 'left' else '右'}侧倾斜"
            
            return {
                "status": status, 
                "score": score, 
                "difference_ratio": difference_ratio,
                "left_distance": left_distance,
                "right_distance": right_distance,
                "lean_direction": lean_direction,
                "description": description
            }
        except Exception as e:
            print(f"评估身体左右倾斜时出错：{str(e)}")
            return {"status": "error", "score": 0, "difference_ratio": None}
    
    def assess_all_postures(self):
        """评估所有体态问题"""
        if not self.keypoints_data:
            print("未加载关键点数据，无法进行评估")
            return False
        
        # 计算平均关键点位置
        avg_keypoints = self._average_keypoints()
        if not avg_keypoints:
            print("计算平均关键点失败")
            return False
        
        # 评估各项体态问题
        self.results["head_forward_tilt"] = self._assess_head_forward_tilt(avg_keypoints)
        self.results["thoracic_kyphosis"] = self._assess_thoracic_kyphosis(avg_keypoints)
        self.results["anterior_pelvic_tilt"] = self._assess_anterior_pelvic_tilt(avg_keypoints)
        self.results["posterior_pelvic_tilt"] = self._assess_posterior_pelvic_tilt(avg_keypoints)
        self.results["lateral_pelvic_tilt"] = self._assess_lateral_pelvic_tilt(avg_keypoints)
        self.results["scoliosis"] = self._assess_scoliosis(avg_keypoints)
        self.results["shoulder_drop"] = self._assess_shoulder_drop(avg_keypoints)
        self.results["knee_valgus"] = self._assess_knee_valgus(avg_keypoints)
        self.results["lateral_leaning"] = self._assess_lateral_leaning(avg_keypoints)
        
        # 计算加权分数
        self._calculate_weighted_score()
        
        return True
    
    def _calculate_weighted_score(self):
        """根据评估结果和权重计算总分"""
        total_score = 0
        total_weight = 0
        
        # 遍历所有评估结果
        for problem, result in self.results.items():
            if problem in POSTURE_WEIGHTS and "score" in result:
                weight = POSTURE_WEIGHTS[problem]
                score = result["score"]
                
                # 记录每个问题的加权分数
                weighted_score = weight * score
                self.scores[problem] = weighted_score
                
                total_score += weighted_score
                total_weight += weight
        
        # 计算总分（满分100）
        if total_weight > 0:
            self.total_score = (total_score / total_weight) * 100
        else:
            self.total_score = 0
        
        return self.total_score
    
    def get_assessment_report(self):
        """生成评估报告"""
        if not self.results:
            return {"error": "未进行评估"}
        
        report = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "total_score": round(self.total_score, 2),
            "details": []
        }
        
        # 添加各项问题的详细评估结果
        for problem, result in self.results.items():
            if problem in POSTURE_WEIGHTS:
                detail = {
                    "problem": problem,
                    "status": result.get("status", "unknown"),
                    "score": result.get("score", 0),
                    "weight": POSTURE_WEIGHTS[problem],
                    "weighted_score": self.scores.get(problem, 0)
                }
                
                # 添加测量数据（如角度）
                if "angle" in result and result["angle"] is not None:
                    detail["measurement"] = {"angle": round(result["angle"], 2)}
                elif "diff_cm" in result and result["diff_cm"] is not None:
                    detail["measurement"] = {"diff_cm": round(result["diff_cm"], 2)}
                elif "tilt_cm" in result and result["tilt_cm"] is not None:
                    detail["measurement"] = {"tilt_cm": round(result["tilt_cm"], 2)}
                elif "gap_mm" in result and result["gap_mm"] is not None:
                    detail["measurement"] = {"gap_mm": round(result["gap_mm"], 2)}
                elif "difference_ratio" in result and result["difference_ratio"] is not None:
                    # 针对左右倾添加更详细的测量数据
                    detail["measurement"] = {
                        "difference_ratio": round(result["difference_ratio"] * 100, 2),  # 转为百分比
                        "left_distance": round(result.get("left_distance", 0), 2),
                        "right_distance": round(result.get("right_distance", 0), 2)
                    }
                    
                    # 添加倾斜方向信息
                    if "lean_direction" in result:
                        detail["lean_direction"] = result["lean_direction"]
                    
                    # 添加描述信息
                    if "description" in result:
                        detail["description"] = result["description"]
                
                report["details"].append(detail)
        
        return report

# 根据需要可以添加更多功能
def get_latest_assessment(data_dir="data/recorded/static"):
    """快速获取最新数据的评估结果"""
    assessor = PostureAssessment()
    if assessor.load_latest_data(data_dir):
        if assessor.assess_all_postures():
            return assessor.get_assessment_report()
    return {"error": "无法完成评估"}

# 测试函数
if __name__ == "__main__":
    assessor = PostureAssessment()
    if assessor.load_latest_data():
        assessor.assess_all_postures()
        report = assessor.get_assessment_report()
        print(json.dumps(report, indent=2, ensure_ascii=False)) 