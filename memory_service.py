import infinity
from infinity import NetworkAddress
from datetime import datetime
import json
from typing import List, Dict, Any, Optional
import aiohttp
import certifi
import ssl
from config import MemoryServiceConfig
import re
import asyncio
import time

import logging

# 配置日志
logging.basicConfig(
    level=logging.ERROR,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def serialize_data(metadata: Optional[Dict[str, Any]], tags: Optional[List[str]]) -> (str, str):
    """序列化 metadata 和 tags 字段为 JSON 字符串"""
    try:
        metadata_str = json.dumps(metadata or {})
    except (TypeError, ValueError) as e:
        logger.warning(f"Failed to serialize metadata: {e}")
        metadata_str = '{}'

    try:
        tags_str = json.dumps(tags or [])
    except (TypeError, ValueError) as e:
        logger.warning(f"Failed to serialize tags: {e}")
        tags_str = '[]'

    return metadata_str, tags_str

class InfinityMemoryService:
    def __init__(self, config: Optional[MemoryServiceConfig] = None):
        """初始化记忆服务"""
        self.config = config or MemoryServiceConfig()
        self._init_connection()
        self._table_cache = {}
        self.ssl_context = ssl.create_default_context(cafile=certifi.where())
        self._closed = False

    def _init_connection(self):
        """初始化数据库连接"""
        max_retries = 3
        retry_delay = 1

        for attempt in range(max_retries):
            try:
                self.infinity_obj = infinity.connect(
                    NetworkAddress(self.config.INFINITY_HOST, self.config.INFINITY_PORT)
                )
                self._init_database()
                return
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    continue
                raise e

    def _reconnect(self):
        """重新连接数据库"""
        try:
            self.infinity_obj.disconnect()
        except:
            pass
        time.sleep(1)  # 等待一秒后重连
        self._init_connection()

    def _ensure_connection(self):
        """确保连接可用"""
        try:
            self.infinity_obj.list_databases()
        except Exception as e:
            logger.warning(f"Connection check failed: {e}")
            self._reconnect()
            # 验证重连
            try:
                self.infinity_obj.list_databases()
            except Exception as e2:
                logger.error(f"Reconnection failed: {e2}")
                raise Exception(f"Failed to reconnect: {e2}")

    def _process_row(self, row) -> Optional[Dict]:
        """处理单行数据"""
        if row is None:
            #logger.error("Received None row in _process_row")
            return None

        try:
            # 处理 polars 数据格式
            memory_id = row['memory_id'] if 'memory_id' in row else None
            if hasattr(memory_id, 'logic_type'):
                return None
            else:
                memory_id = memory_id[0]

            content = str(row['content'][0]) if 'content' in row else None
            timestamp = str(row['timestamp'][0]) if 'timestamp' in row else None

            # 处理 metadata
            metadata_str = str(row['metadata'][0]) if 'metadata' in row else '{}'
            metadata = {}
            if metadata_str.strip() and metadata_str != 'null':
                try:
                    metadata = json.loads(metadata_str)
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse meta {metadata_str}: {e}")

            # 处理 tags
            tags_str = str(row['tags'][0]) if 'tags' in row else '[]'
            tags = []
            if tags_str.strip() and tags_str != 'null':
                try:
                    tags = json.loads(tags_str)
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse tags: {tags_str}: {e}")

            # 处理 score
            score = float(row['_score'][0]) if '_score' in row else None

            return {
                "memory_id": memory_id,
                "content": content,
                "timestamp": timestamp,
                "metadata": metadata,
                "tags": tags,
                "score": score
            }

        except Exception as e:
            logger.error(f"Error processing row: {e} , row: {row}", exc_info=True)
            return None

    def _init_database(self):
        """初始化数据库"""
        try:
            # 首先检查数据库是否存在
            databases = self.infinity_obj.list_databases()
            database_exists = False
            # 使用 db_names 字段检查数据库是否存在
            if hasattr(databases, 'db_names'):
                database_exists = self.config.DEFAULT_DATABASE in databases.db_names
            if not database_exists:
                # 数据库不存在时创建
                self.infinity_obj.create_database(self.config.DEFAULT_DATABASE)
            # 获取数据库连接
            self.db = self.infinity_obj.get_database(self.config.DEFAULT_DATABASE)
        except infinity.common.InfinityException as e:
            if "Duplicated db entry" in str(e):
                # 如果数据库已存在，直接获取连接
                self.db = self.infinity_obj.get_database(self.config.DEFAULT_DATABASE)
            else:
                # 其他错误则抛出
                raise e

    def _get_table_name(self, tenant_id: str, project_id: str) -> str:
        """生成表名"""
        # 确保tenant_id和project_id只包含安全的字符
        safe_tenant_id = re.sub(r'[^a-zA-Z0-9_]', '_', tenant_id)
        safe_project_id = re.sub(r'[^a-zA-Z0-9_]', '_', project_id)
        return f"{self.config.TABLE_PREFIX}{safe_tenant_id}_{safe_project_id}"

    def _get_table(self, tenant_id: str, project_id: str):
        """获取或创建表"""
        table_name = self._get_table_name(tenant_id, project_id)
        if table_name not in self._table_cache:
            try:
                # 根据不同的模型设置正确的向量类型
                vector_type = "double" if self.config.EMBEDDING_SERVICE_TYPE.lower() == "ollama" else "float"
                # 尝试创建新表
                table = self.db.create_table(
                    table_name,
                    {
                        "memory_id": {"type": "varchar"},
                        "content": {"type": "varchar"},
                        "embedding": {"type": f"vector,{self.config.EMBEDDING_DIM},{vector_type}"},  # 向量类型
                        "timestamp": {"type": "timestamp"},
                        "metadata": {"type": "varchar"},
                        "tags": {"type": "varchar"}
                    }
                )
                # 创建索引
                table.create_index(
                    f"{table_name}_embedding_idx",
                    ["embedding"],
                    index_type="hnsw",
                    params={
                        "M": self.config.HNSW_M,
                        "ef_construction": self.config.HNSW_EF_CONSTRUCTION
                    }
                )
                table.create_index(
                    f"{table_name}_content_idx",
                    ["content"],
                    index_type="fulltext",
                    params={"analyzer": "standard"}
                )
            except Exception as e:
                # 如果表已存在，获取现有表
                table = self.db.get_table(table_name)
            self._table_cache[table_name] = table
        return self._table_cache[table_name]

    async def _get_embedding(self, text: str) -> List[float]:
        """调用外部向量化服务获取文本的向量表示"""
        try:
            if self.config.EMBEDDING_SERVICE_TYPE.lower() == "ollama":
                # Ollama API 调用
                payload = {
                    "model": self.config.EMBEDDING_MODEL,
                    "prompt": text
                }
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                            f"{self.config.EMBEDDING_SERVICE_URL}/api/embeddings",
                            headers={"Content-Type": "application/json"},
                            json=payload,
                            ssl=False if "localhost" in self.config.EMBEDDING_SERVICE_URL else self.ssl_context
                    ) as response:
                        if response.status != 200:
                            raise Exception(f"Ollama embedding service error: {await response.text()}")
                        result = await response.json()
                        # 确保返回的向量维度正确
                        embedding = result['embedding']
                        if len(embedding) != self.config.EMBEDDING_DIM:
                            raise ValueError(
                                f"Embedding dimension mismatch. Expected {self.config.EMBEDDING_DIM}, got {len(embedding)}")
                        # 将向量值转换为 double 类型
                        return [float(x) for x in embedding]
            else:
                # OpenAI API 调用
                headers = {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.config.EMBEDDING_API_KEY}"
                }
                payload = {
                    "input": text,
                    "model": self.config.EMBEDDING_MODEL
                }
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                            self.config.EMBEDDING_SERVICE_URL,
                            headers=headers,
                            json=payload,
                            ssl=self.ssl_context
                    ) as response:
                        if response.status != 200:
                            raise Exception(f"OpenAI embedding service error: {await response.text()}")
                        result = await response.json()
                        return result['data'][0]['embedding']

        except Exception as e:
            logger.error(f"Error in _get_embedding: {str(e)}")
            raise

    async def add_memory(self,
                         tenant_id: str,
                         project_id: str,
                         content: str,
                         metadata: Dict[str, Any] = None,
                         tags: List[str] = None) -> str:
        """添加新记忆"""
        table = self._get_table(tenant_id, project_id)
        embedding = await self._get_embedding(content)
        memory_id = f"mem_{tenant_id}_{project_id}_{datetime.now().strftime('%Y%m%d%H%M%S%f')}"
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        # 使用 serialize_data 函数
        metadata_str, tags_str = serialize_data(metadata, tags)

        table.insert([{
            "memory_id": memory_id,
            "content": content,
            "embedding": embedding,
            "timestamp": timestamp,
            "metadata": metadata_str,
            "tags": tags_str
        }])

        logger.debug(f"Inserting memory with metadata: {metadata_str} and tags: {tags_str}")

        return memory_id

    async def batch_add_memories(self,
                                 tenant_id: str,
                                 project_id: str,
                                 memories: List[Dict[str, Any]]) -> List[str]:
        """批量添加记忆"""
        table = self._get_table(tenant_id, project_id)
        memory_ids = []
        batch_size = 10  # 减小批次大小

        try:
            for i in range(0, len(memories), batch_size):
                batch = memories[i:i + batch_size]
                batch_data = []

                # 并行获取embeddings
                embedding_tasks = [
                    self._get_embedding(memory['content'])
                    for memory in batch
                ]

                embeddings = await asyncio.gather(*embedding_tasks)

                for j, (memory, embedding) in enumerate(zip(batch, embeddings)):
                    memory_id = f"mem_{tenant_id}_{project_id}_{datetime.now().strftime('%Y%m%d%H%M%S%f')}_{i}_{j}"
                    memory_ids.append(memory_id)

                    # 使用 serialize_data 函数
                    metadata_str, tags_str = serialize_data(memory.get('metadata', {}), memory.get('tags', []))

                    batch_data.append({
                        "memory_id": memory_id,
                        "content": memory['content'],
                        "embedding": embedding,
                        "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        "metadata": metadata_str,
                        "tags": tags_str
                    })
                    logger.debug(f"batch_add_memories: Inserting memory with metadata: {metadata_str} and tags: {tags_str}")

                table.insert(batch_data)
                await asyncio.sleep(0.1)  # 添加短暂延迟

            return memory_ids

        except Exception as e:
            logger.error(f"Error in batch insert: {e}")
            return []

    async def get_memory(self,
                         tenant_id: str,
                         project_id: str,
                         memory_id: str) -> Optional[Dict]:
        """获取指定记忆"""
        self._ensure_connection()
        table = self._get_table(tenant_id, project_id)
        try:
            result = table.filter(f"memory_id = '{memory_id}'").output(["*"]).to_result()
            logger.debug(f"Retrieved data: {result}")
            if result and len(result) > 0:
                processed_row = self._process_row(result[0])
                return processed_row if processed_row else None
            return None
        except Exception as e:
            print(f"Error retrieving memory: {e}")
            return None

    async def list_memories(self,
                            tenant_id: str,
                            project_id: str,
                            limit: int = 10,
                            offset: int = 0) -> List[Dict]:
        """列出所有记忆"""
        self._ensure_connection()
        table = self._get_table(tenant_id, project_id)
        try:
            results = table.output(["*"]).limit(limit).offset(offset).to_pl()
            memories = []

            if results and len(results) > 0:
                for row in results:
                    logger.debug(f"List_memories: Retrieved data: {row}")
                    processed_row = self._process_row(row)
                    if processed_row:
                        memories.append(processed_row)
            return memories
        except Exception as e:
            print(f"Error listing memories: {e}")
            return []

    async def search_memory(self,
                            tenant_id: str,
                            project_id: str,
                            query_text: Optional[str] = None,
                            filter_metadata: Optional[Dict[str, Any]] = None,
                            filter_tags: Optional[List[str]] = None,
                            limit: int = 10) -> List[Dict]:
        """搜索记忆"""
        self._ensure_connection()
        table = self._get_table(tenant_id, project_id)

        try:
            if query_text:
                # 获取查询文本的向量表示
                query_embedding = await self._get_embedding(query_text)

                # 优先使用向量搜索
                search_query = table.match_dense(
                    vector_column_name="embedding",
                    embedding_data=query_embedding,
                    embedding_data_type="float",
                    distance_type="ip",
                    topn=limit
                )

                # 执行向量搜索
                results = search_query.output(["memory_id", "content", "timestamp", "metadata", "tags"]).to_result()

                # 如果向量搜索没有结果，尝试文本搜索
                if len(results) == 0:
                    text_query = table.match_text(
                        fields="content",
                        matching_text=query_text,
                        topn=limit
                    )
                    results = text_query.output(["memory_id", "content", "timestamp", "metadata", "tags"]).to_result()
            else:
                # 如果没有查询文本，直接返回最新记录
                results = table.output(["memory_id", "content", "timestamp", "metadata", "tags"]).limit(
                    limit).to_result()

            # 处理结果
            memories = []
            if results is not None and len(results) > 0:
                for idx in range(len(results)):
                    try:
                        row = results[idx]
                        if not row:
                            continue

                        processed_row = self._process_row(row)
                        if processed_row:
                            # 在内存中进行过滤
                            if filter_metadata:
                                metadata_match = all(
                                    str(processed_row['metadata'].get(k)) == str(v)
                                    for k, v in filter_metadata.items()
                                )
                                if not metadata_match:
                                    continue

                            if filter_tags:
                                # 确保所有过滤标签都存在于记忆的标签中
                                memory_tags = set(processed_row['tags'])
                                filter_tags_set = set(filter_tags)
                                if not filter_tags_set.issubset(memory_tags):
                                    continue

                            memories.append(processed_row)

                    except Exception as row_error:
                        logger.error(f"Error processing row {idx}: {row_error}")
                        continue

            return memories[:limit]  # 确保不超过限制数量

        except Exception as e:
            logger.error(f"Error searching memories: {e}", exc_info=True)
            return []

    async def update_memory(self,
                            tenant_id: str,
                            project_id: str,
                            memory_id: str,
                            content: Optional[str] = None,
                            metadata: Optional[Dict[str, Any]] = None,
                            tags: Optional[List[str]] = None) -> bool:
        """更新记忆"""
        self._ensure_connection()
        table = self._get_table(tenant_id, project_id)

        try:
            update_data = {}

            if content is not None:
                embedding = await self._get_embedding(content)
                update_data.update({
                    "content": content,
                    "embedding": embedding
                })

            if metadata is not None or tags is not None:
                metadata_str, tags_str = serialize_data(metadata, tags)
                update_data["metadata"] = metadata_str
                update_data["tags"] = tags_str

            if update_data:
                table.update(
                    cond=f"memory_id = '{memory_id}'",
                    data=update_data
                )

            return True

        except Exception as e:
            logger.error(f"Error updating memory: {e}")
            return False

    async def delete_memory(self,
                            tenant_id: str,
                            project_id: str,
                            memory_id: str) -> bool:
        """删除指定记忆"""
        table = self._get_table(tenant_id, project_id)
        try:
            table.filter(f"memory_id = '{memory_id}'").delete()
            return True
        except Exception as e:
            print(f"Error deleting memory: {e}")
            return False

    def close(self):
        """显式关闭连接"""
        if not self._closed:
            try:
                self.infinity_obj.disconnect()
                self._closed = True
            except Exception as e:
                print(f"Warning: Error during disconnect: {e}")

    def __enter__(self):
        """支持上下文管理器"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """支持上下文管理器"""
        self.close()

    async def __aenter__(self):
        """异步上下文管理器入口"""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """异步上下文管理器出口"""
        await self.close()

    async def close(self):
        """异步关闭连接"""
        if not hasattr(self, '_closed') or not self._closed:
            try:
                self.infinity_obj.disconnect()
                self._closed = True
            except Exception as e:
                print(f"Warning: Error during disconnect: {e}")