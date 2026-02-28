"""动态历史表：按接口名创建 {interface}_history 表"""
from datetime import datetime
from sqlalchemy import Column, DateTime, Integer, String, Table, Text
from sqlalchemy.dialects.postgresql import JSONB
from .db import Base, engine


def get_history_table(interface_name: str) -> Table:
    """获取或创建历史表对象，注册到 Base.metadata"""
    table_name = f"{interface_name}_history"
    if table_name in Base.metadata.tables:
        return Base.metadata.tables[table_name]
    table = Table(
        table_name,
        Base.metadata,
        Column("id", Integer, primary_key=True),
        Column("created_at", DateTime(timezone=True), nullable=False, default=datetime.utcnow),
        Column("user_id", String(64), nullable=True),
        Column("request_text", Text, nullable=False),
        Column("response_text", Text, nullable=False),
        Column("meta", JSONB, nullable=True),
        extend_existing=True,
    )
    return table


async def ensure_history_table(interface_name: str) -> Table:
    """确保数据库中存在该历史表"""
    table = get_history_table(interface_name)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all, tables=[table])
    return table
