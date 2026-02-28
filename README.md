# Agentic RAG — 基于商业研究报告的万金油式智能问答与报告生成系统

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.109+-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

**万金油式多模态文档处理 | 向量检索 | Agent 自主规划 | 可溯源报告生成**

[功能特性](#功能特性) · [创新点](#创新点) · [技术架构](#技术架构) · [快速开始](#快速开始) · [API 文档](#api-文档)

</div>

---

## 个人心得
在因为该项目要通过fastapi部署，因此项目内的导包最好用绝对路径到就是说从该项目的根目录就开始导，不要用相对路径，比如.file_processor或者..qdrant_manager。这种在部署时极易发生导包错误，比如某个包未被声明，某个方法未被定义等等。本人就因为这个浪费了大量时间，因此为了稳定，只要使用fastapi的项目导包一律从项目根目录开始。
这个项目本人觉得最精彩的地方就是给文件切片入库的方法了。这里简述一下:把文本类型的文件例如.ppt,.pdf,.docx等等都转化为图片形式，然后用大模型生成每张图片的摘要，把摘要作为检索向量；用ocr技术提取每张图片上的全部内容最为引用内容。最后都写入向量数据库。这种方法太万金油了，能保证任何情况都能取得不错的效果。
还一个就是根据不同的情况使用智能体进行检索。系统会对传入的文件进行评判：要是传入的文件是一个非常专业，非常难懂的报告，此时系统就会自动调用智能体来进行检索；要是传入的文件只是一个比较简单的，那么系统就会调用普通的rag。这样的目的就是控制成本，毕竟token太贵了。




## 项目简介

Agentic RAG 是一个面向**商业研究报告**的智能问答与报告生成系统。系统支持 PDF、DOCX、PPTX 及多种图片格式的文档上传，通过多模态摘要、OCR、向量化构建知识库，提供 **Base RAG**（单次检索问答）与 **Agentic RAG**（Agent 多轮检索、自主规划、生成报告）两种模式。

### 功能特性

| 功能 | 描述 |
|------|------|
| **文档入库** | ZIP 批量上传，支持 PDF/DOCX/PPTX/图片，中文文件名自动解码 |
| **多模态处理** | Qwen-VL 图片摘要 + Kimi OCR 全文提取 + Qwen Embedding 向量化 |
| **Base RAG** | 单次向量检索 + DeepSeek 生成精简回答 |
| **Agentic RAG** | Agent 自主多轮检索、问题拆解、信息综合，生成结构化报告并输出 TXT |
| **历史记录** | 按接口动态建表，支持分页、按用户过滤 |
| **可溯源** | 回答基于检索内容，标注来源，禁止臆测与编造 |

---

## 创新点

### 1. Agentic RAG vs 传统 RAG
*这么设计的原因*：成本至至上，能够最大限度的省token，在控制成本的同时最大化提高检索精度。现在大模型的token越来越贵，要是每个检索都用智能体来搞那会太贵了，于是有选择的使用agent。
- **传统 RAG**：用户提问 → 单次检索 → 一次性生成回答，难以覆盖多角度、跨文档的综合问题
- **本系统 Agentic RAG**：
  - Agent 根据问题类型判断是否调用 RAG
  - 支持多角度子问题、回溯问题，**最多 5 次** 工具调用
  - 引入**收敛条件**（证据充分、边际收益过低、资料不足、调用上限）防止无限检索
  - 输出前自检：是否基于检索内容、是否有无依据断言、是否满足收敛

### 2. 多模态文档流水线
*这么设计的原因*：万金油方法，不管什么类型，什么领域的数据，用这个方法都能达到一个不错的水平，并且也省token。要是有钱的话你可以不用ocr，直接让大模型给你提取图片上所有的信息，效果更好
- 文档统一转换为 PDF → 每页转图片
- 图片经 **Qwen-VL 摘要**（200 字以内）形成语义向量，用于检索
- **Kimi OCR** 提取全文作为最终引用内容
- 摘要向量化后写入 Qdrant，检索时返回 origin_text，实现「摘要检索 + 原文引用」

### 3. 动态历史表设计

- 按接口名动态创建 `{interface}_history` 表
- 新接口只需在启动时补充一行 `ensure_history_table("new_api")`
- 表结构统一，支持 JSONB meta 扩展

### 4. 双模式设计

- **Base RAG**：低延迟、直接回答，适合简单问答
- **Agentic RAG**：高精度、多轮推理，适合复杂分析与报告

---

## 技术架构

### 技术栈

| 层级 | 技术 | 用途 |
|------|------|------|
| Web | FastAPI | REST API |
| LLM | DeepSeek Chat | 对话与报告生成 |
| 多模态 | Qwen-VL-max | 图片摘要 |
| Embedding | Qwen text-embedding-v4 (1024 维) | 向量化 |
| 向量库 | Qdrant | 向量存储与检索 |
| OCR | Kimi (Moonshot API) | 图片文字识别 |
| Agent | DeepAgents (LangChain/LangGraph) | 工具调用与规划 |
| 文档 | PyMuPDF、python-docx、python-pptx、docx2pdf | 解析与转换 |
| 数据库 | PostgreSQL + SQLAlchemy (async) | 历史记录 |

### 项目结构

```
agentic-rag/
├── api.py                 # FastAPI 入口
├── config.py              # 环境变量与配置
├── llm_factory.py         # 大模型与 Embedding 封装
├── qdrant_manager.py      # Qdrant 向量库管理
├── file_processor.py      # 文档/图片处理流水线
├── tools/
│   └── base_rag.py        # RAG 检索与生成
├── prompt/
│   └── agentic_report_prompt.py  # Agent 系统提示词
├── database/
│   ├── db.py              # 数据库连接
│   ├── history_tables.py  # 动态历史表
│   └── history_repository.py  # 历史读写
├── report_output/         # Agent 报告输出目录
├── requirements.txt
├── .env.example
└── README.md
```

### 数据流

```
文档入库：ZIP → 解压 → 转 PDF → 转图 → 摘要 + OCR → 向量化 → Qdrant
Base RAG：用户问题 → 向量检索 → 拼接检索内容 → DeepSeek 生成回答
Agentic RAG：用户问题 → Agent → 多轮 search_base_rag → 归纳 → 收敛 → 生成报告 → 写 TXT
```

---

## 快速开始

### 1. 环境要求

- Python 3.10+
- PostgreSQL
- Qdrant（本地或远程）

### 2. 安装依赖

```bash
git clone https://github.com/YOUR_USERNAME/Agentic_Rag.git
cd Agentic_Rag
pip install -r requirements.txt
```

### 3. 配置环境变量

```bash
cp .env.example .env
# 编辑 .env，填入 DASHSCOPE_API_KEY、DEEPSEEK_API_KEY、KIMI_API_KEY、
# DATABASE_URL、QDRANT_URL、QDRANT_COLLECTION
```

### 4. 启动服务

```bash
uvicorn api:app --reload --host 0.0.0.0 --port 8000
```

访问 http://localhost:8000/docs 查看 Swagger 文档。

### 5. 使用流程

1. **上传文档**：`POST /upload/zip` 上传 ZIP，系统解析并写入 Qdrant
2. **Base RAG**：`POST /rag/base?message=艾力斯公司2024年业绩如何` 获取单次问答
3. **Agentic RAG**：`POST /rag/agentic?message=艾力斯公司2024年突破汇总报告` 获取 Agent 生成的报告，并可在 `report_output/` 中查看 TXT
4. **历史记录**：`GET /rag/base/history` 或 `GET /rag/agentic/history` 查询历史

---

## API 文档

| 方法 | 路径 | 描述 |
|------|------|------|
| GET | / | 健康检查 |
| POST | /upload/zip | 上传 ZIP，解析文档并入库 |
| POST | /rag/base | Base RAG 问答（query: message） |
| GET | /rag/base/history | Base RAG 历史（可选 limit, offset, user_id） |
| POST | /rag/agentic | Agentic RAG 报告生成（query: message） |
| GET | /rag/agentic/history | Agentic RAG 历史 |

---

## 开发能力体现

- **后端开发**：FastAPI 异步 API、结构化项目与依赖管理  
- **大模型应用**：多 Provider 封装、RAG、Agent 工具调用、Prompt 工程  
- **多模态处理**：图片摘要、OCR、文档解析、格式转换  
- **向量检索**：Qdrant 集成、Embedding、相似度检索  
- **数据库**：PostgreSQL 异步 ORM、动态表设计、SQLAlchemy Core  
- **工程实践**：环境变量管理、异常处理、临时文件清理、依赖注入  

---

## License

MIT
