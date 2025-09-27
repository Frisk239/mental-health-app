# 心灵健康助手 (Mental Health Assistant)

基于AI的多模态大学生心理健康支持系统，整合面部识别、语音分析、文本挖掘，实现情绪监测、游戏化干预和隐私保护。

## 🚀 核心功能

### 多模态情绪感知
- **面部表情识别**: 实时分析面部微表情
- **语音情绪分析**: 识别语调中的情绪变化
- **文本情感挖掘**: 分析社交媒体文字情绪

### 游戏化心理调节
- **情绪农场**: 作物生长对应情绪状态
- **社交实验室**: AI陪伴的社交技能练习

### 隐私计算干预
- **联邦学习**: 本地训练，保护数据隐私
- **个性化建议**: 基于用户历史生成干预方案

## 🛠️ 技术栈

### 前端
- **React 18** + **TypeScript**
- **TensorFlow.js** - 浏览器端AI推理
- **Phaser.js** - HTML5游戏引擎
- **Tailwind CSS** - 响应式设计

### 后端
- **FastAPI** (Python) - 高性能API
- **PyTorch** + **Transformers** - AI模型
- **PostgreSQL** + **Redis** - 数据存储

### AI模型
- **MediaPipe Face Mesh** - 面部识别
- **Wav2Vec2** - 语音识别
- **BERT** - 文本情感分析

## 📦 快速开始

### 1. 环境准备

```bash
# 安装Node.js 18+ 和 Python 3.9+
# 克隆项目
git clone https://github.com/Frisk239/mental-health-app.git
cd mental-health-app
```

### 2. 配置环境变量

```bash
# 复制环境变量模板
cp .env.example .env

# 编辑 .env 文件，配置以下必需项：
# - DEEPSEEK_API_KEY: 从 https://platform.deepseek.com/ 获取
# - DATABASE_URL: 默认使用SQLite，无需修改
# - 其他配置可保持默认值
```

### 3. 安装依赖

```bash
# 前端依赖
cd frontend
npm install

# 后端依赖 (推荐使用conda)
cd ../backend
conda create -n mental-health python=3.9
conda activate mental-health
pip install -r requirements.txt
```

### 4. 初始化数据库

```bash
# 初始化SQLite数据库
cd backend
python scripts/init_db.py
```

### 4. 启动服务

```bash
# 启动后端API
cd backend
python main.py

# 启动前端开发服务器
cd frontend
npm run dev
```

访问 http://localhost:3000 开始使用！

## 🏗️ 项目结构

```
mental-health-app/
├── frontend/                 # React前端应用
│   ├── src/
│   │   ├── components/       # 可复用组件
│   │   ├── pages/           # 页面组件
│   │   ├── hooks/           # 自定义hooks
│   │   ├── services/        # API服务
│   │   └── types/           # TypeScript类型
│   ├── public/              # 静态资源
│   └── package.json
├── backend/                 # FastAPI后端
│   ├── app/
│   │   ├── routes/          # API路由
│   │   ├── models/          # 数据模型
│   │   ├── services/        # 业务逻辑
│   │   └── utils/           # 工具函数
│   ├── main.py             # 应用入口
│   └── requirements.txt
├── shared/                  # 共享代码
├── docs/                   # 文档
└── .env                    # 环境变量
```

## 🔧 开发指南

### 前端开发

```bash
cd frontend
npm run dev          # 启动开发服务器
npm run build        # 构建生产版本
npm run lint         # 代码检查
```

### 后端开发

```bash
cd backend
python main.py       # 启动API服务器
pytest               # 运行测试
black .             # 代码格式化
```

## 📊 API文档

启动后端服务后，访问：
- **API文档**: http://localhost:8000/docs
- **健康检查**: http://localhost:8000/health

## 🔒 隐私保护

- **本地优先**: 敏感数据存储在用户设备
- **联邦学习**: 模型训练不上传原始数据
- **加密传输**: 所有API调用使用HTTPS
- **合规设计**: 符合GDPR隐私保护标准

## 🤝 贡献指南

1. Fork项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 创建Pull Request

## 📄 许可证

本项目采用MIT许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 🙏 致谢

- [TalkHeal](https://github.com/eccentriccoder01/TalkHeal) - 对话系统参考
- [MentalHealth](https://github.com/galihru/MentalHealth) - 面部识别实现
- [Multimodal-Emotion-Recognition](https://github.com/maelfabien/Multimodal-Emotion-Recognition) - 多模态pipeline
- [Emotion-LLaMA](https://github.com/ZebangCheng/Emotion-LLaMA) - 情感推理模型

## 📞 联系我们

如有问题或建议，请提交Issue或联系开发团队。

---

**让AI成为心理健康的得力助手！** 💙
