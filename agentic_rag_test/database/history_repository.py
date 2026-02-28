"""历史记录写入与读取"""
from typing import Any, Dict, List, Optional
from sqlalchemy import select
from .db import AsyncSessionLocal
from .history_tables import ensure_history_table


async def log_history(
    interface_name: str,
    request_text: str,
    response_text: str,
    user_id: Optional[str] = None,
    meta: Optional[Dict[str, Any]] = None,
) -> int:
    """插入历史记录，返回主键 id"""
    table = await ensure_history_table(interface_name)
    async with AsyncSessionLocal() as session:
        async with session.begin():
            result = await session.execute(
                table.insert().values(
                    user_id=user_id,
                    request_text=request_text,
                    response_text=response_text,
                    meta=meta,
                )
            )
            return result.inserted_primary_key[0]


async def get_history(
    interface_name: str,
    limit: int = 20,
    offset: int = 0,
    user_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """按时间倒序分页查询历史，支持 user_id 过滤"""
    table = await ensure_history_table(interface_name)
    stmt = select(table).order_by(table.c.created_at.desc()).offset(offset).limit(limit)
    if user_id:
        stmt = stmt.where(table.c.user_id == user_id)
    async with AsyncSessionLocal() as session:
        rows = (await session.execute(stmt)).mappings().all()
    return [
        {
            "id": row["id"],
            "created_at": row["created_at"].isoformat(),
            "user_id": row["user_id"],
            "request_text": row["request_text"],
            "response_text": row["response_text"],
            "meta": row["meta"],
        }
        for row in rows
    ]
