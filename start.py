"""
姿态分析与步态检测系统启动脚本
"""

import os
import sys
import argparse
import subprocess
import time
import webbrowser
import socket
import platform
import requests
from requests.exceptions import ConnectionError
import cv2
import torch
import multiprocessing as mp
import numpy as np

from core.utils import SharedData
import config

def check_port_available(port):
    """检查端口是否可用"""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    result = False
    try:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("0.0.0.0", port))
        result = True
    except socket.error:
        print(f"[警告] 端口 {port} 已被占用，请尝试其他端口")
    finally:
        sock.close()
    return result

def check_requirements():
    """检查必要的依赖是否已安装"""
    required_packages = ["flask", "numpy", "torch", "requests", "opencv-python"]
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == "opencv-python":
                __import__("cv2")
            else:
                __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"[错误] 缺少以下依赖包: {', '.join(missing_packages)}")
        install = input("是否自动安装这些依赖? (y/n): ")
        if install.lower() == 'y':
            subprocess.call([sys.executable, "-m", "pip", "install", *missing_packages])
            print("[信息] 依赖安装完成")
        else:
            print("[信息] 请手动安装依赖后再启动系统")
            return False
            
    return True

def check_server_ready(url, max_retries=20, retry_interval=1):
    """检查服务器是否已准备好接受请求"""
    print(f"等待服务启动完成...")
    for i in range(max_retries):
        try:
            response = requests.get(url, timeout=2)
            if response.status_code == 200:
                print(f"服务已启动完成，耗时 {i * retry_interval} 秒")
                return True
        except ConnectionError:
            pass
        except Exception as e:
            print(f"检查服务状态时出错: {str(e)}")
        
        time.sleep(retry_interval)
        sys.stdout.write(".")
        sys.stdout.flush()
    
    print(f"\n[警告] 服务启动超时，已尝试 {max_retries * retry_interval} 秒")
    return False

def start_web_server(args):
    """启动Web服务器"""
    # 检查是否存在必要的文件
    if not os.path.exists('flask_app.py'):
        print("[错误] 未找到主程序文件 flask_app.py")
        return False
        
    if not os.path.exists('templates/index.html'):
        print("[错误] 未找到模板文件 templates/index.html")
        return False
    
    # 检查端口是否可用
    if not check_port_available(args.port):
        return False
    
    # 构建命令行参数
    cmd = [sys.executable, 'flask_app.py']
    
    if args.config:
        cmd.extend(['--config', args.config])
    
    if args.model:
        cmd.extend(['--model', args.model])
    
    if args.device:
        cmd.extend(['--device', args.device])
    
    cmd.extend(['--port', str(args.port)])
    cmd.extend(['--host', args.host])
    
    if args.debug:
        cmd.append('--debug')
    
    # 输出启动信息
    print("=" * 60)
    print("姿态分析与步态检测系统")
    print("=" * 60)
    print(f"主机: {args.host}")
    print(f"端口: {args.port}")
    if args.device:
        print(f"设备: {args.device}")
    if args.model:
        print(f"模型: {args.model}")
    print("=" * 60)
    print("正在启动Web服务...")
    
    # 启动系统
    try:
        # 如果不是Windows系统，则设置新进程组，使得Ctrl+C只影响启动脚本
        if platform.system() != 'Windows':
            p = subprocess.Popen(cmd, preexec_fn=os.setsid)
        else:
            p = subprocess.Popen(cmd)
        
        # 自动打开浏览器
        if not args.no_browser:
            # 使用localhost替代0.0.0.0作为访问地址
            browser_host = "localhost" if args.host == "0.0.0.0" else args.host
            url = f"http://{browser_host}:{args.port}"
            
            if args.wait_ready:
                # 等待服务器完全启动
                server_ready = check_server_ready(url)
                if server_ready:
                    print(f"在浏览器中打开: {url}")
                    webbrowser.open(url)
                else:
                    print(f"服务启动超时，请手动在浏览器中访问: {url}")
            else:
                # 原来的行为：等待固定时间后打开浏览器
                time.sleep(3)
                print(f"在浏览器中打开: {url}")
                webbrowser.open(url)
        
        return p
    except Exception as e:
        print(f"启动Web服务过程中出错: {str(e)}")
        return None

def run_camera_system(args):
    """运行多摄像头姿态估计系统"""
    try:
        # 如果提供了配置文件，则加载配置
        if args.config and os.path.exists(args.config):
            user_config = config.ConfigManager.load_config(args.config)
            print(f"已加载配置文件: {args.config}")
        
        # 如果命令行指定了调试模式，则更新配置
        if args.debug:
            config.SystemConfig.DEBUG_MODE = True
            print("调试模式已开启")
            
        # 检测可用设备
        device = args.device if args.device else ('cuda:0' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {device}")
        
        # 创建多进程管理器
        manager = mp.Manager()
        return_dict = manager.dict()
        
        # 创建共享数据结构
        shared_data = SharedData()
        shared_data.running.value = True
        
        from multiprocess_camera import camera_process
        
        # 获取模型名称
        model_name = args.model if args.model else config.ModelConfig.DEFAULT_MODEL
        
        # 创建两个摄像头处理进程
        process1 = mp.Process(target=camera_process, args=(0, return_dict, shared_data, model_name, device))
        process2 = mp.Process(target=camera_process, args=(1, return_dict, shared_data, model_name, device))
        
        # 启动进程
        process1.start()
        process2.start()
        
        print("摄像头处理已启动。按'q'键退出")
        
        while True:
            # 检查是否有帧需要显示
            frames_to_show = False
            
            if 'frame_0' in return_dict:
                # 解码图像数据
                buffer = np.frombuffer(return_dict['frame_0'], dtype=np.uint8)
                display_frame1 = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
                cv2.imshow('CAM0 - RTMPose', display_frame1)
                frames_to_show = True
                
            if 'frame_1' in return_dict:
                # 解码图像数据
                buffer = np.frombuffer(return_dict['frame_1'], dtype=np.uint8)
                display_frame2 = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
                cv2.imshow('CAM1 - RTMPose', display_frame2)
                frames_to_show = True
                
            # 检查错误
            for cam_id in [0, 1]:
                if f'error_{cam_id}' in return_dict:
                    print(f"CAM {cam_id} Error: {return_dict[f'error_{cam_id}']}")
                    return_dict.pop(f'error_{cam_id}', None)
            
            # 检查退出 - 只在有帧显示时处理键盘事件
            if frames_to_show and cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        return shared_data, process1, process2
    except Exception as e:
        print(f"摄像头处理错误: {str(e)}")
        return None, None, None

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='姿态分析与步态检测系统启动脚本')
    parser.add_argument('--config', type=str, help='配置文件路径')
    parser.add_argument('--model', type=str, help='模型名称')
    parser.add_argument('--device', type=str, help='设备名称 (cpu/cuda:0)')
    parser.add_argument('--port', type=int, default=5000, help='Web服务端口号')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Web服务主机地址')
    parser.add_argument('--no-browser', action='store_true', help='不自动打开浏览器')
    parser.add_argument('--debug', action='store_true', help='启用调试模式')
    parser.add_argument('--wait-ready', action='store_true', default=True, help='等待服务完全就绪后再打开浏览器')
    parser.add_argument('--mode', type=str, choices=['web', 'camera', 'all'], default='web', help='运行模式：web服务器，摄像头处理，或全部')
    
    args = parser.parse_args()
    
    # 检查依赖
    if not check_requirements():
        return 1
    
    web_process = None
    shared_data = None
    camera_process1 = None
    camera_process2 = None
    
    try:
        # 根据模式启动相应服务
        if args.mode in ['web', 'all']:
            web_process = start_web_server(args)
            if not web_process:
                return 1
        
        if args.mode in ['camera', 'all']:
            # 设置多进程启动方法
            if args.mode == 'camera':  # 如果同时运行web服务和摄像头，不要重复设置
                mp.set_start_method('spawn', force=True)
            shared_data, camera_process1, camera_process2 = run_camera_system(args)
        
        # 等待进程完成
        if args.mode == 'web' and web_process:
            print("Web服务已启动。按 Ctrl+C 停止...")
            web_process.wait()
        elif args.mode == 'all':
            print("系统已完全启动。按 Ctrl+C 停止...")
            while True:
                time.sleep(1)
                
    except KeyboardInterrupt:
        print("正在停止系统...")
    
    except Exception as e:
        print(f"启动过程中出错: {str(e)}")
        return 1
    
    finally:
        # 清理资源
        if shared_data:
            shared_data.running.value = False
            
        # 等待摄像头进程结束
        if camera_process1 and camera_process1.is_alive():
            camera_process1.join(timeout=1.0)
            if camera_process1.is_alive():
                camera_process1.terminate()
                
        if camera_process2 and camera_process2.is_alive():
            camera_process2.join(timeout=1.0)
            if camera_process2.is_alive():
                camera_process2.terminate()
        
        # 关闭窗口
        cv2.destroyAllWindows()
        
        # 停止Web服务
        if web_process:
            web_process.terminate()
            
        print("系统已停止")
    
    return 0

if __name__ == "__main__":
    # 如果运行摄像头模式，设置多进程启动方法
    if len(sys.argv) > 1 and '--mode' in sys.argv:
        mode_index = sys.argv.index('--mode')
        if mode_index + 1 < len(sys.argv) and sys.argv[mode_index + 1] in ['camera', 'all']:
            mp.set_start_method('spawn', force=True)
    sys.exit(main()) 