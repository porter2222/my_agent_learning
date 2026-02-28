"""
Qdrant 向量数据库管理 - 向量存储与相似度检索
"""
import uuid
from typing import List, Dict, Any

from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, Distance, VectorParams

from config import QDRANT_URL, QDRANT_COLLECTION


class QDRANT_MANAGER:
    """Qdrant 向量库管理，负责集合初始化、向量存储与检索"""

    def __init__(self):
        self.client = QdrantClient(url=QDRANT_URL)
        self.collection_name = QDRANT_COLLECTION
        self._init_collection()

    def _init_collection(self):
        """初始化集合，若不存在则创建（1024 维，余弦相似度）"""
        try:
            collections = self.client.get_collections().collections
            exists = any(c.name == self.collection_name for c in collections)
            if not exists:
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=1024,  # Qwen text-embedding-v4 维度
                        distance=Distance.COSINE,
                    ),
                )
                print(f"[INFO] 创建集合: {self.collection_name}")
        except Exception as e:
            raise Exception(f"初始化 Qdrant 集合失败: {e}") from e

    def store_vectors(self, vector: List[float], metadata: Dict[str, Any]) -> str:
        """存储单条向量及元数据，返回向量 ID"""
        vector_id = str(uuid.uuid4())
        point = PointStruct(id=vector_id, vector=vector, payload=metadata)
        self.client.upsert(
            collection_name=self.collection_name,
            points=[point],
        )
        return vector_id

    def search_vectors(
        self,
        query_vector: List[float],
        limit: int = 12,
        score_threshold: float = 0.1,
    ) -> list:
        """按余弦相似度检索，返回相似向量列表（含 payload）"""
        return self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            with_payload=True,
            limit=limit,
            score_threshold=score_threshold,
        ).points
