#!/usr/bin/env python
# -*- coding: utf-8 -*-

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
def check_port_available(port):
    """检查端口是否可用"""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    result = False
    try:
        sock.bind(("0.0.0.0", port))
        result = True
    except:
        print(f"[警告] 端口 {port} 已被占用，请尝试其他端口")
    finally:
        sock.close()
    return result

def check_requirements():
    """检查必要的依赖是否已安装"""
    required_packages = ["flask", "numpy", "torch"]
    missing_packages = []
    
    for package in required_packages:
        try:
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
    
    args = parser.parse_args()
    
    # 检查是否存在必要的文件
    if not os.path.exists('flask_app.py'):
        print("[错误] 未找到主程序文件 flask_app.py")
        return 1
        
    if not os.path.exists('templates/index.html'):
        print("[错误] 未找到模板文件 templates/index.html")
        return 1
    
    # 检查依赖
    if not check_requirements():
        return 1
    
    # 检查端口是否可用
    if not check_port_available(args.port):
        return 1
    
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
    
    # 创建日志管理器
  
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
    print("正在启动系统...")
    
    # 启动系统
    try:
        # 如果不是Windows系统，则设置新进程组，使得Ctrl+C只影响启动脚本
        if platform.system() != 'Windows':
            p = subprocess.Popen(cmd, preexec_fn=os.setsid)
        else:
            p = subprocess.Popen(cmd)
        
        # 等待服务启动
        time.sleep(3)
        
        # 自动打开浏览器
        if not args.no_browser:
            # 使用localhost替代0.0.0.0作为访问地址
            browser_host = "localhost" if args.host == "0.0.0.0" else args.host
            url = f"http://{browser_host}:{args.port}"
            print(f"在浏览器中打开: {url}")
            webbrowser.open(url)
        
        print("系统已启动。按 Ctrl+C 停止...")
        p.wait()
        
    except KeyboardInterrupt:
        print("系统已停止")
    
    except Exception as e:
        print(f"启动过程中出错: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 