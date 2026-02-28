"""
Agentic RAG - FastAPI 入口
"""
import os
import io
import zipfile
from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from agentic_rag_test.agentic_rag.file_processor import FileProcessor  # 文档处理类
from deepagents import create_deep_agent
from langchain.agents import create_agent
from dotenv import load_dotenv
from langchain_deepseek import ChatDeepSeek
from agentic_rag_test.agentic_rag.prompt.agentic_report_prompt import SYSTEM_PROMPT
from agentic_rag_test.agentic_rag.tools.base_rag import ask_base_rag,search_base_rag
from agentic_rag_test.agentic_rag.database.history_repository import log_history, get_history
from agentic_rag_test.agentic_rag.database.history_tables import ensure_history_table
from agentic_rag_test.agentic_rag.database.db import engine, Base
from agentic_rag_test.agentic_rag.database import models # noqa: F401  # 注册 ORM 模型，便于 create_all
from typing import Optional
from agentic_rag_test.agentic_rag.qdrant_manager import QDRANT_MANAGER
from deepagents.backends import FilesystemBackend

load_dotenv()

# DeepSeek 模型（Agent 推理）
deepseek_model = ChatDeepSeek(
    model="deepseek-chat",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

# Qdrant 管理实例（共享给 FileProcessor 和 base_rag）
qdrant_manager = QDRANT_MANAGER()

app = FastAPI(
    title="Agentic RAG",
    description="基于商业研究报告的智能问答与报告生成系统",
    version="1.0.0",
)


@app.on_event("startup")
async def on_startup() -> None:
    """启动时创建历史表"""
    for iface in ("rag_base", "rag_agentic"):
        await ensure_history_table(iface)


@app.get("/")
def root():
    """健康检查"""
    return {"message": "Agentic RAG is running", "docs": "/docs"}


@app.post("/upload/zip")
async def upload_zip(file: UploadFile = File(...)):
    """上传 ZIP，解压并处理文档/图片，向量化后存入 Qdrant"""
    try:
        processor = FileProcessor(qdrant_manager=qdrant_manager)
        zip_content = await file.read()
        with zipfile.ZipFile(io.BytesIO(zip_content)) as zf:
            decoded_files = []
            decode_map = {}
            for raw_name in zf.namelist():
                decoded_name = raw_name
                try:
                    decoded_name = raw_name.encode("cp437").decode("gbk")
                except Exception:
                    try:
                        decoded_name = raw_name.encode("cp437").decode("gb2312")
                    except Exception:
                        pass
                decoded_files.append(decoded_name)
                decode_map[decoded_name] = raw_name

            doc_exts = (".pdf", ".docx", ".pptx", ".doc")
            img_exts = (".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp", ".tiff", ".tif")
            target_files = [
                f for f in decoded_files
                if f.lower().endswith(doc_exts + img_exts)
            ]
            if not target_files:
                raise HTTPException(
                    status_code=400,
                    detail="未找到可处理的文档或图片文件",
                )

            results = []
            for decoded_name in target_files:
                try:
                    raw_name = decode_map[decoded_name]
                    file_bytes = zf.read(raw_name)
                    ext = os.path.splitext(decoded_name)[1].lower()
                    if ext in img_exts:
                        processor.process_image_file(file_bytes, decoded_name)
                    else:
                        processor.process_file_content(file_bytes, decoded_name)
                    results.append({"filename": decoded_name, "status": "success"})
                except Exception as e:
                    results.append({
                        "filename": decoded_name,
                        "status": "failed",
                        "error": str(e),
                    })
            processor.cleanup()
        return {"message": "文件处理完成", "results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/rag/base")
async def base_rag(message: str, request: Request):
    """Base RAG：单次检索 + 生成回答"""
    ai_response = ask_base_rag(message)
    await log_history(
        "rag_base",
        request_text=message,
        response_text=ai_response,
        user_id=None,
        meta={"endpoint": "/rag/base"},
    )
    return ai_response


@app.get("/rag/base/history")
async def base_rag_history(
    limit: int = 20,
    offset: int = 0,
    user_id: Optional[str] = None,
):
    """Base RAG 历史记录"""
    return await get_history("rag_base", limit=limit, offset=offset, user_id=user_id)


@app.post("/rag/agentic")
async def agentic_rag(message: str, request: Request):
    """Agentic RAG：Agent 多轮检索、综合信息、生成报告"""
    agent = create_deep_agent(
        model=deepseek_model,
        tools=[search_base_rag],
        system_prompt=SYSTEM_PROMPT,
        backend=FilesystemBackend(root_dir="./report_output", virtual_mode=True),
        debug=True,
    )
    result = agent.invoke({"messages": [{"role": "user", "content": message}]})
    content = result["messages"][-1].content
    await log_history(
        "rag_agentic",
        request_text=message,
        response_text=content,
        user_id=None,
        meta={"endpoint": "/rag/agentic"},
    )
    return content


@app.get("/rag/agentic/history")
async def agentic_rag_history(
    limit: int = 20,
    offset: int = 0,
    user_id: Optional[str] = None,
):
    """Agentic RAG 历史记录"""
    return await get_history(
        "rag_agentic", limit=limit, offset=offset, user_id=user_id
    )
