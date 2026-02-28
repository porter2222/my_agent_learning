"""
大模型与 Embedding 统一封装 - 支持 Qwen、DeepSeek、OpenAI 等多种 provider
"""
import os
import json
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


class LLMClient:
    """统一封装多种大模型和 Embedding 服务"""

    def __init__(self, provider: str, model: str = None):
        self.provider = provider
        self.client, self.model = self._get_client_and_model(provider, model)

    def _get_client_and_model(self, provider: str, model: str = None):
        if provider == "qwen-cn":
            return (
                OpenAI(
                    api_key=os.getenv("DASHSCOPE_API_KEY"),
                    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
                ),
                model or "qwen-plus",
            )
        elif provider == "deepseek":
            return (
                OpenAI(
                    api_key=os.getenv("DEEPSEEK_API_KEY"),
                    base_url="https://api.deepseek.com",
                ),
                model or "deepseek-chat",
            )
        elif provider == "openai":
            return (
                OpenAI(
                    api_key=os.getenv("OPENAI_API_KEY"),
                    base_url="https://api.openai.com/v1",
                ),
                model or "gpt-4o-mini",
            )
        else:
            raise ValueError(f"未知 provider: {provider}")

    def chat(self, message: str) -> str:
        """非流式对话，返回完整回复"""
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": message},
            ],
            stream=False,
        )
        return resp.choices[0].message.content

    def qwen_vision(self, image_data: str, prompt: str) -> str:
        """多模态调用 - 图片 + 文本，用于摘要生成"""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{image_data}"},
                        },
                    ],
                }
            ],
        )
        return response.choices[0].message.content

    def embedding(self, message) -> list:
        """文本向量化，返回 1024 维向量（Qwen embedding）"""
        completion = self.client.embeddings.create(
            model=self.model,
            input=message,
        )
        return json.loads(completion.model_dump_json())["data"][0]["embedding"]
