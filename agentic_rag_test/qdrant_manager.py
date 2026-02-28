"""
Qdrant 向量数据库管理 - 向量存储与相似度检索
"""
import uuid
from typing import List, Dict, Any, Optional

from qdrant_client import QdrantClient
from qdrant_client.grpc import ScoredPoint
from qdrant_client.models import Distance, VectorParams
from qdrant_client.models import PointStruct, PointIdsList

from agentic_rag_test.agentic_rag.config import QDRANT_URL, QDRANT_COLLECTION
from agentic_rag_test.agentic_rag.llm_factory import LLMClient


class QDRANT_MANAGER:
    def __init__(self):
        """初始化 Qdrant 客户端"""
        self.client = QdrantClient(url=QDRANT_URL)
        self.collection_name = QDRANT_COLLECTION
        self._init_collection()

    def _init_collection(self):
        """初始化集合，如果不存在则创建"""
        try:
            # 检查集合是否存在
            collections = self.client.get_collections().collections
            exists = any(collection.name == self.collection_name for collection in collections)

            if not exists:
                # 创建新的集合
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=1024,  # Qwen embedding 维度
                        distance=Distance.COSINE
                    )
                )
                print(f"创建新的集合: {self.collection_name}")
        except Exception as e:
            raise Exception(f"初始化 Qdrant 集合失败: {str(e)}")

    def store_vectors(self, vector: List[float], metadata: Dict[str, Any]) -> str:
        """
        存储单条已生成的向量到 Qdrant

        Args:
            vector: 已经生成好的向量（来自 process_single_image）
            metadata: 元数据（image_path, origin_text, page 等）

        Returns:
            vector_id: Qdrant 存储 ID
        """

        try:
            # 生成向量 ID
            vector_id = str(uuid.uuid4())

            # 构造向量点
            point = PointStruct(
                id=vector_id,
                vector=vector,  # ← 向量由上游多线程阶段生成
                payload=metadata
            )

            # 写入 Qdrant
            result = self.client.upsert(
                collection_name=self.collection_name,
                points=[point]
            )
            print("插入成功",result)
            return vector_id

        except Exception as e:
            raise Exception(f"存储向量失败: {str(e)}")

    def store_vectors_bulk(self, vectors: List[List[float]], metadatas: List[Dict[str, Any]]) -> List[str]:
        """
        批量存储多条向量到 Qdrant（单个文件处理完后一次 upsert）

        Args:
            vectors: 向量列表
            metadatas: 与向量对应的 metadata 列表

        Returns:
            List[str]: 写入的向量 ID 列表
        """
        try:
            if len(vectors) != len(metadatas):
                raise ValueError("vectors 与 metadatas 数量不一致")

            points = []
            ids = []
            for vector, metadata in zip(vectors, metadatas):
                vector_id = str(uuid.uuid4())
                print(metadata)
                print(len(vector))
                ids.append(vector_id)
                points.append(PointStruct(
                    id=vector_id,
                    vector=vector,
                    payload=metadata
                ))
            print("待插入数据",len(points))

            result = self.client.upsert(
                collection_name=self.collection_name,
                wait=True,
                points=points
            )
            print("插入数据成功",result)
            return ids
        except Exception as e:
            raise Exception(f"批量存储向量失败: {str(e)}")

    def search_vectors(self,
                       query_vector: List[float],
                       limit: int = 12,
                       score_threshold: float = 0.1) -> List[ScoredPoint]:
        """
        搜索相似向量

        Args:
            query_vector: 查询向量
            limit: 返回结果数量
            score_threshold: 相似度阈值

        Returns:
            List[score_threshold]: 搜索结果列表
        """
        try:
            search_result = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                with_payload=True,
                limit=limit,
                score_threshold=score_threshold
            ).points

            return search_result
        except Exception as e:
            raise Exception(f"搜索向量失败: {str(e)}")

    def delete_vectors(self, vector_ids: List[str]) -> bool:
        """
        删除指定的向量

        Args:
            vector_ids: 要删除的向量ID列表

        Returns:
            bool: 是否删除成功
        """
        try:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=PointIdsList(
                    points=vector_ids
                )
            )
            return True
        except Exception as e:
            raise Exception(f"删除向量失败: {str(e)}")

    def get_vector_by_id(self, vector_id: str) -> Optional[Dict[str, Any]]:
        """
        根据ID获取向量

        Args:
            vector_id: 向量ID

        Returns:
            Optional[Dict]: 向量信息，如果不存在则返回None
        """
        try:
            result = self.client.retrieve(
                collection_name=self.collection_name,
                ids=[vector_id]
            )
            if result and len(result) > 0:
                point = result[0]
                return {
                    'vector': point.vector,
                    'payload': point.payload
                }
            return None
        except Exception as e:
            raise Exception(f"获取向量失败: {str(e)}")

    def update_vector_metadata(self,
                               vector_id: str,
                               metadata: Dict[str, Any]) -> bool:
        """
        更新向量的元数据

        Args:
            vector_id: 向量ID
            metadata: 新的元数据

        Returns:
            bool: 是否更新成功
        """
        try:
            self.client.set_payload(
                collection_name=self.collection_name,
                payload=metadata,
                points=[vector_id]
            )
            return True
        except Exception as e:
            raise Exception(f"更新向量元数据失败: {str(e)}")

    def get_collection_stats(self) -> Dict[str, Any]:
        """
        获取集合统计信息

        Returns:
            Dict: 统计信息
        """
        try:
            collection_info = self.client.get_collection(self.collection_name)
            return {
                'vectors_count': collection_info.vectors_count,
                'points_count': collection_info.points_count,
                'segments_count': collection_info.segments_count,
                'config': collection_info.config
            }
        except Exception as e:
            raise Exception(f"获取集合统计信息失败: {str(e)}")

if __name__ == "__main__":
    qdrant_manager = QDRANT_MANAGER()
    qwen_embedding = LLMClient(provider="qwen-cn", model="text-embedding-v4")
    query_vector = qwen_embedding.embedding("艾力斯公司2024年发生了什么事情")
    print(query_vector)

    metadata1 = {
        "filename": "你好",  # ← 用你想要的中文名
        "page": 1,
        "image_path": "path1/sds",
        "origin_text": "你好好的u份额u无法v我饿u研发v一",
        "summary_text": "一段文件"
    }
    metadata2 = {
        "filename": "decoded_name2",  # ← 用你想要的中文名
        "page": 2,
        "image_path": "path2",
        "origin_text": "text2",
        "summary_text": "summary2"
    }
    qwen_embedding = LLMClient(provider="qwen-cn", model="text-embedding-v4")
    vector = qwen_embedding.embedding("summary1")
    print(qdrant_manager.store_vectors(vector, metadata1))
