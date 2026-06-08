# 🏀 HoopCut

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue?style=for-the-badge&logo=python)
![YOLOv8](https://img.shields.io/badge/YOLOv8-AI%20Detection-green?style=for-the-badge)
![Flask](https://img.shields.io/badge/Flask-Backend-red?style=for-the-badge&logo=flask)

**🎯 基于AI的篮球视频自动剪辑系统**

使用本地 YOLOv8 模型实现篮球进球自动检测和视频集锦生成。视频、模型推理、临时文件和输出结果都保存在本机。

</div>

## 🎬 演示视频
<video src="./assets/result-demo.mp4" controls width="720">
你的浏览器不支持 video 标签。可点击链接直接下载查看。
</video>


https://github.com/user-attachments/assets/8723aabc-38b1-4c8e-90a3-13d688a820bf

## 🧩整体pipeline

<img width="1589" height="867" alt="屏幕截图 2025-12-16 231438" src="https://github.com/user-attachments/assets/6b113e21-f470-477b-b85f-d6520e7dd992" />


## ✨ 核心功能

- 🎯 **AI进球检测**：基于YOLOv8模型的篮球进球自动识别
- 🎬 **视频集锦生成**：FFmpeg自动剪辑生成精彩进球集锦
- 📊 **统计分析**：提供投篮统计和命中率分析
- 🚀 **REST API**：完整的后端API接口服务
- ⚡ **实时处理**：支持视频上传和实时处理进度反馈

## 🛠️ 技术栈

- **Flask** - Python Web框架
- **YOLOv8** (Ultralytics) - AI目标检测模型
- **OpenCV** - 计算机视觉库
- **FFmpeg** - 视频处理工具

## 📋 系统要求

- **Python 3.8+**
- **Node.js 16+**
- **FFmpeg** (用于视频处理)
- **支持的视频格式**：MP4, AVI, MOV, MKV, WebM, FLV, WMV

## 🖥️ 本地处理模式

HoopCut 当前按完全本地模式运行：

- 前端地址：`http://127.0.0.1:5173`
- 后端 API：默认 `http://127.0.0.1:5050` 起，会自动选择一个可用本地端口
- 本地模型：`backend/best.pt`
- 上传目录：`backend/uploads/`
- 临时目录：`backend/temp/`
- 输出目录：`backend/outputs/`

后端只监听 `127.0.0.1`，不会把处理服务暴露到局域网。

## 🚀 快速开始

```bash
# 克隆项目
git clone <repository-url>
cd HoopCut

# 自动创建 backend/venv、安装依赖，并选择一个可用后端端口
./start-local.sh
```

### 手动启动

```bash
# 启动后端
cd backend

python3 -m venv venv

source venv/bin/activate

pip install -r requirements.txt
BACKEND_PORT=5050 FRONTEND_PORT=5173 venv/bin/python app.py

# 新开终端启动前端
cd frontend

npm install
VITE_PROXY_TARGET=http://127.0.0.1:5050 \
VITE_API_BASE_URL= \
VITE_SOCKET_URL= \
npm run dev -- --host 127.0.0.1 --port 5173

```

## 前端页面展示

<img width="2557" height="1055" alt="image" src="https://github.com/user-attachments/assets/fd98d4f4-fee8-4d09-95f5-c65ac9587bf5" />

## 处理过程页面

<img width="2549" height="1247" alt="image" src="https://github.com/user-attachments/assets/617a48ed-3cf8-47d9-899f-c69b2920e541" />

## 进球判断逻辑

<img width="1551" height="805" alt="屏幕截图 2025-12-16 231541" src="https://github.com/user-attachments/assets/4c312c84-b87f-477f-afb8-52021b03ec8e" />

<img width="1587" height="778" alt="image" src="https://github.com/user-attachments/assets/982d31da-49cf-4e1f-bf6c-be8c6e9cce69" />



## 📁 项目结构

```
HoopCut/
├── backend/                 # 后端代码
│   ├── app.py              # 主应用文件
│   ├── shot_detector_video.py  # 进球检测模块
│   ├── video_processor.py   # 视频处理模块
│   ├── utils.py            # 工具函数
│   ├── requirements.txt    # 依赖配置
│   ├── uploads/            # 上传文件目录
│   ├── outputs/            # 输出文件目录
│   └── AI-Basketball-Shot-Detection-Tracker/  # AI模型
├── models/                 # 模型文件目录
├── outputs/                # 全局输出目录
├── uploads/                # 全局上传目录
└── README.md              # 项目说明文档
```

## 📡 API接口

### 分块上传初始化
```bash
POST /api/upload/init
Content-Type: application/json

{
  "filename": "demo.mp4"
}
```

### 上传分块
```bash
POST /api/upload/chunk
Content-Type: multipart/form-data

# 参数
- chunk: 视频分块
- fileId: 上传初始化返回的文件 ID
- chunkIndex: 分块序号，从 0 开始
```

### 完成分块上传
```bash
POST /api/upload/complete
Content-Type: application/json

{
  "fileId": "...",
  "filename": "demo.mp4",
  "totalChunks": 12
}
```

### 启动本地处理
```bash
POST /api/process
Content-Type: application/json

{
  "fileId": "...",
  "beforeSeconds": 3,
  "afterSeconds": 1
}
```

### 获取处理状态
```bash
GET /api/progress/{task_id}
```

### 播放集锦视频
```bash
GET /api/stream/{filename}
```

### 下载集锦视频
```bash
GET /api/download/{filename}
```

## 📄 许可证

MIT License


## 🙏 致谢
- 本项目受 [AI-Basketball-Shot-Detection-Tracker](https://github.com/avishah3/AI-Basketball-Shot-Detection-Tracker) 的启发，感谢作者提供的思路与开源贡献。







**🏀 帮你发现篮球场上的每一个精彩瞬间！**
