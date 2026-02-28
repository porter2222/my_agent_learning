"""
RAG 检索与生成 - Base RAG 与 Agent 工具 search_base_rag
"""
from qdrant_manager import QDRANT_MANAGER
from llm_factory import LLMClient

qdrant_manager = QDRANT_MANAGER()
qwen_embedding = LLMClient(provider="qwen-cn", model="text-embedding-v4")


def search_base_rag(message: str) -> str:
    """
    【商业深度报告专业检索与分析工具】
    从商业研究报告知识库中检索与用户问题相关的内容，返回拼接的检索结果文本。
    适用场景：公司研究、行业分析、市场趋势、商业模式、研究结论等需基于资料回答的问题。
    不适用：日常常识、纯主观判断、无需资料支撑的简短问题。
    输入：message (str) 用户问题；输出：拼接的「来源文件 + 切片内容」文本。
    """
    message_vector = qwen_embedding.embedding(message)
    result = qdrant_manager.search_vectors(message_vector)
    search_data = ""
    for scp in result:
        image_path = scp.payload.get("image_path", "")
        origin_text = scp.payload.get("origin_text", "")
        search_data += f"来源文件：\n{origin_text}\n切片内容：\n{image_path}\n\n"
    return search_data


def ask_base_rag(message: str) -> str:
    """Base RAG 问答：检索 + 大模型生成"""
    search_data = search_base_rag(message)
    rag_prompt = f"""
根据以下内容，结合用户的问题，给出详细且专业的回答，回答时请引用来源文件中的内容：
用户问题：{message}
检索内容：{search_data}
请基于以上内容回答，仅输出答案，精简专业。
"""
    deepseek_chat = LLMClient(provider="deepseek", model="deepseek-chat")
    return deepseek_chat.chat(rag_prompt)
