from flask import Flask, render_template, Response, jsonify, request, session
import cv2
import numpy as np
import time
import random
from datetime import datetime
from flask_cors import CORS  # 导入CORS扩展

app = Flask(__name__)
CORS(app)  # 启用CORS支持
app.secret_key = "mmpose_posture_analysis_secret_key"  # 用于session加密

# 全局变量保存当前状态
global_state = {
    "is_recording": False,
    "recording_type": None,  # "static" 或 "gait"
    "keypoints_count": 0,
    "frames_count": 0,
    "camera_errors": {0: 0, 1: 0},
    "saved_dir": None,
    "assessment_result": None  # 保存体态评估结果
}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/posture_report')
def posture_report():
    """体态报告页面"""
    return render_template('posture_report.html')

@app.route('/video_feed/<int:camera_id>')
def video_feed(camera_id):
    """视频流路由，返回摄像头画面"""
    # 在实际应用中，这里应当返回真实的视频流
    # 示例代码使用假数据
    return Response(
        generate_frames(camera_id), 
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

def generate_frames(camera_id):
    """生成视频帧的函数，这里应当连接到实际的视频源"""
    # 实际项目中应替换为真实视频帧生成逻辑
    
    # 模拟视频流
    while True:
        try:
            # 创建一个模拟的帧
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            
            # 添加一些文字和图形，模拟摄像头画面
            cv2.putText(frame, f"Camera {camera_id}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(frame, f"Time: {time.strftime('%H:%M:%S')}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # 绘制一个移动的圆，模拟动态内容
            t = time.time() * 2
            x = int(320 + 200 * np.sin(t))
            y = int(240 + 150 * np.cos(t))
            cv2.circle(frame, (x, y), 20, (0, 165, 255), -1)
            
            # 如果正在记录，添加指示器
            if global_state["is_recording"]:
                cv2.circle(frame, (30, 30), 10, (0, 0, 255), -1)  # 红色圆圈表示录制中
                cv2.putText(frame, "Recording", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # 更新关键点和帧计数（模拟）
                global_state["keypoints_count"] += 1
                global_state["frames_count"] += 1
            
            # 编码帧为JPEG格式
            ret, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            
            # 返回帧
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            
            # 控制帧率
            time.sleep(0.05)  # 约20 FPS
            
        except Exception as e:
            print(f"Error generating frame for camera {camera_id}: {e}")
            global_state["camera_errors"][camera_id] += 1
            time.sleep(0.5)  # 出错时短暂暂停

@app.route('/get_data_status')
def get_data_status():
    """获取当前数据收集状态"""
    return jsonify({
        "is_recording": global_state["is_recording"],
        "recording_type": global_state["recording_type"],
        "keypoints_count": global_state["keypoints_count"],
        "frames_count": global_state["frames_count"],
        "camera_errors": global_state["camera_errors"]
    })

@app.route('/start_static_pose_analysis', methods=['POST'])
def start_static_pose_analysis():
    """开始静态体态分析"""
    global_state["is_recording"] = True
    global_state["recording_type"] = "static"
    global_state["keypoints_count"] = 0
    global_state["frames_count"] = 0
    
    # 实际项目中这里应当启动真实的数据收集
    
    return jsonify({
        "status": "success",
        "message": "开始静态体态数据收集"
    })

@app.route('/start_gait_analysis', methods=['POST'])
def start_gait_analysis():
    """开始步态分析"""
    global_state["is_recording"] = True
    global_state["recording_type"] = "gait"
    global_state["keypoints_count"] = 0
    global_state["frames_count"] = 0
    
    # 实际项目中这里应当启动真实的数据收集
    
    return jsonify({
        "status": "success",
        "message": "开始步态数据收集"
    })

@app.route('/complete_analysis', methods=['POST'])
def complete_analysis():
    """完成分析并保存数据"""
    global_state["is_recording"] = False
    saved_dir = "/data/posture_analysis/session_" + str(int(time.time()))
    global_state["saved_dir"] = saved_dir
    
    # 实际项目中这里应当保存真实的收集数据
    
    return jsonify({
        "status": "success",
        "message": "数据已成功保存",
        "saved_dir": saved_dir
    })

@app.route('/analyze_posture', methods=['POST'])
def analyze_posture():
    """后台处理体态分析（实际计算）"""
    print("收到体态分析请求")  # 调试信息
    try:
        # 在实际项目中，这里应该执行真正的体态分析计算
        # 这里模拟一个分析结果
        
        # 模拟处理时间
        time.sleep(2)
        
        # 生成模拟的体态评估结果
        assessment_result = {
            "status": "success",
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "total_score": random.randint(40, 90),
            "details": [
                {
                    "problem": "head_forward_tilt",
                    "status": random.choice(["normal", "mild", "moderate", "severe"]),
                    "score": random.randint(0, 100),
                    "weight": 1.5,
                    "weighted_score": 0,  # 将在下面计算
                    "measurement": {"angle": random.uniform(10, 45)}
                },
                {
                    "problem": "thoracic_kyphosis",
                    "status": random.choice(["normal", "mild", "moderate"]),
                    "score": random.randint(0, 100),
                    "weight": 1.2,
                    "weighted_score": 0,
                    "measurement": {"angle": random.uniform(20, 60)}
                },
                {
                    "problem": "shoulder_drop",
                    "status": random.choice(["normal", "mild"]),
                    "score": random.randint(0, 100),
                    "weight": 1.0,
                    "weighted_score": 0,
                    "measurement": {"diff_cm": random.uniform(0, 4)}
                },
                {
                    "problem": "scoliosis",
                    "status": random.choice(["normal", "mild", "moderate", "severe"]),
                    "score": random.randint(0, 100),
                    "weight": 1.8,
                    "weighted_score": 0,
                    "measurement": {"angle": random.uniform(0, 20)}
                },
                {
                    "problem": "anterior_pelvic_tilt",
                    "status": random.choice(["normal", "mild", "moderate"]),
                    "score": random.randint(0, 100),
                    "weight": 1.2,
                    "weighted_score": 0,
                    "measurement": {"angle": random.uniform(5, 25)}
                }
            ]
        }
        
        # 计算加权分数
        for detail in assessment_result["details"]:
            detail["weighted_score"] = round(detail["score"] * detail["weight"])
        
        # 保存结果到全局状态
        global_state["assessment_result"] = assessment_result
        
        print("体态分析计算完成，返回成功响应")  # 调试信息
        return jsonify({
            "status": "success",
            "message": "体态分析计算完成"
        })
    except Exception as e:
        print(f"体态分析计算出错: {e}")  # 调试信息
        import traceback
        traceback.print_exc()  # 打印详细错误信息
        return jsonify({
            "status": "error",
            "message": str(e)
        })

@app.route('/get_assessment_data', methods=['GET'])
def get_assessment_data():
    """获取体态评估数据，供报告页面使用"""
    if global_state["assessment_result"] is None:
        return jsonify({
            "status": "error",
            "message": "未找到体态评估数据，请先进行体态分析"
        })
    
    return jsonify({
        "status": "success",
        "result": global_state["assessment_result"]
    })

@app.route('/reset_camera/<int:camera_id>', methods=['POST'])
def reset_camera(camera_id):
    """重置摄像头"""
    # 在实际项目中，这里应当重置真实摄像头
    
    return jsonify({
        "status": "success",
        "message": f"摄像头 {camera_id} 已重置"
    })

@app.route('/test_api', methods=['GET', 'POST'])
def test_api():
    """测试API是否正常工作"""
    return jsonify({
        "status": "success",
        "message": "API测试成功",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000) 