"""
深度记忆聊天机器人 - 生产级完备版
修复已知错误，增强完整性、健壮性与资源管理
"""
import os
import re
import time
import json
import logging
import threading
import hashlib
from typing import List, Dict, Optional, Callable, Any
from collections import deque

# 可选依赖
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import chromadb
from chromadb.utils import embedding_functions
import openai
import tiktoken

# ----------------------------- 日志配置 -----------------------------
logger = logging.getLogger("DeepMemoryBot")
logger.setLevel(logging.DEBUG)

# ----------------------------- 全局工具 -----------------------------
def get_encoding():
    """使用 cl100k_base 编码（适用于 gpt-3.5-turbo / text-embedding-ada-002 等）"""
    return tiktoken.get_encoding("cl100k_base")

# ----------------------------- 配置管理 -----------------------------
class Config:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-ada-002")
    CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-3.5-turbo")
    COLLECTION_NAME = os.getenv("COLLECTION_NAME", "deep_memory_chatbot")
    CHUNK_MAX_TOKENS = int(os.getenv("CHUNK_MAX_TOKENS", "300"))
    CHUNK_OVERLAP_TOKENS = int(os.getenv("CHUNK_OVERLAP_TOKENS", "50"))
    SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.92"))
    CONFLICT_SIMILARITY = float(os.getenv("CONFLICT_SIMILARITY", "0.85"))
    VERIFY_CONFIDENCE_THRESHOLD = float(os.getenv("VERIFY_CONFIDENCE_THRESHOLD", "0.8"))
    DIALOG_MAX_TOKENS = int(os.getenv("DIALOG_MAX_TOKENS", "1000"))
    RETRY_MAX_ATTEMPTS = int(os.getenv("RETRY_MAX_ATTEMPTS", "3"))
    RETRY_BASE_DELAY = float(os.getenv("RETRY_BASE_DELAY", "1.0"))
    DB_PATH = os.getenv("DB_PATH", "./deep_memory_db")
    MAX_LEARN_TEXT_LENGTH = 50000
    MIN_REPLACE_CONFIDENCE = float(os.getenv("MIN_REPLACE_CONFIDENCE", "0.75"))
    SHORT_TEXT_MIN_CONFIDENCE = float(os.getenv("SHORT_TEXT_MIN_CONFIDENCE", "0.9"))
    # 对话记忆最大保留轮次（以防内存泄漏）
    DIALOG_MAX_TURNS = int(os.getenv("DIALOG_MAX_TURNS", "200"))
    # 源可信度权重：用于修正事实核查置信度
    SOURCE_WEIGHTS = {
        "user_command": 1.1,
        "file_learn": 1.0,
        "chat_auto": 0.95,
        "user_correction": 1.2
    }
    # 自动学习预过滤黑名单
    AUTO_LEARN_BLACKLIST = [
        r'\b开玩笑\b', r'\b假设\b', r'\b如果\b.*\b就好\b', r'\b哈哈\b',
        r'\b测试\b', r'\btest\b', r'\b不知道\b', r'\b随便\b'
    ]
    EXPORT_FILENAME = "memory_export_{timestamp}.json"

    @classmethod
    def validate(cls):
        if not cls.OPENAI_API_KEY:
            raise ValueError("❌ OPENAI_API_KEY 未设置")
        logger.info("配置验证通过")

# ----------------------------- LLM 客户端 -----------------------------
class LLMClient:
    def __init__(self, api_key: str = None):
        self.api_key = api_key or Config.OPENAI_API_KEY
        openai.api_key = self.api_key

    def chat_completion(self, messages, model=None, temperature=0.5, max_tokens=600):
        model = model or Config.CHAT_MODEL
        resp = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        return resp.choices[0].message.content.strip()

# ----------------------------- 重试机制 -----------------------------
def api_retry(func: Callable, *args, **kwargs):
    """
    对可能的瞬时 API 错误进行指数退避重试，
    仅捕获常见服务端错误，不拦截键盘中断等异常。
    """
    for attempt in range(Config.RETRY_MAX_ATTEMPTS):
        try:
            return func(*args, **kwargs)
        except (openai.error.RateLimitError,
                openai.error.ServiceUnavailableError,
                openai.error.APIError,
                openai.error.APIConnectionError,
                openai.error.Timeout) as e:
            if attempt == Config.RETRY_MAX_ATTEMPTS - 1:
                raise
            delay = Config.RETRY_BASE_DELAY * (2 ** attempt)
            logger.warning(f"API 调用失败，{delay:.1f}s 后重试 (尝试 {attempt+1}/{Config.RETRY_MAX_ATTEMPTS}): {e}")
            time.sleep(delay)
        except Exception:
            # 非瞬态错误立刻抛出
            raise

# ----------------------------- 文本处理 -----------------------------
class TextProcessor:
    @staticmethod
    def extract_text_from_file(file_path: str) -> str:
        ext = os.path.splitext(file_path)[1].lower()
        if ext in ('.txt', '.md'):
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        elif ext == '.pdf':
            try:
                import pypdf
                reader = pypdf.PdfReader(file_path)
            except ImportError:
                try:
                    import PyPDF2
                    reader = PyPDF2.PdfReader(file_path)
                except ImportError:
                    raise ImportError("需要安装 pypdf 或 PyPDF2")
            # 防止 None 内容
            return '\n'.join(page.extract_text() or '' for page in reader.pages)
        elif ext == '.docx':
            try:
                from docx import Document
            except ImportError:
                raise ImportError("需要安装 python-docx")
            doc = Document(file_path)
            return '\n'.join(p.text for p in doc.paragraphs)
        else:
            raise ValueError(f"不支持的文件格式：{ext}")

    @staticmethod
    def split_sentences(text: str) -> List[str]:
        text = re.sub(r'([。！？!?.])\s*', r'\1\n', text)
        text = re.sub(r'\n+', '\n', text)
        return [s.strip() for s in text.split('\n') if s.strip()]

    @staticmethod
    def chunk_text_by_tokens(text: str, max_tokens: int, overlap_tokens: int) -> List[str]:
        enc = get_encoding()
        tokens = enc.encode(text)
        chunks = []
        start = 0
        while start < len(tokens):
            end = start + max_tokens
            chunk_tokens = tokens[start:end]
            chunks.append(enc.decode(chunk_tokens))
            if end >= len(tokens):
                break
            start = end - overlap_tokens
            if start <= 0:
                start = 1
        return chunks

    @classmethod
    def chunk_text_semantically(cls, text: str) -> List[str]:
        enc = get_encoding()
        sentences = cls.split_sentences(text)
        chunks = []
        current_sents = []
        current_tokens = 0
        max_tokens = Config.CHUNK_MAX_TOKENS
        overlap_limit = Config.CHUNK_OVERLAP_TOKENS

        for sent in sentences:
            sent_tokens = len(enc.encode(sent))
            if sent_tokens > max_tokens:
                if current_sents:
                    chunks.append(''.join(current_sents))
                    current_sents = []
                    current_tokens = 0
                sub_chunks = cls.chunk_text_by_tokens(sent, max_tokens, overlap_limit)
                chunks.extend(sub_chunks)
                continue

            if current_tokens + sent_tokens > max_tokens and current_sents:
                chunks.append(''.join(current_sents))
                overlap_sents = []
                overlap_tok_count = 0
                for s in reversed(current_sents):
                    t = len(enc.encode(s))
                    if overlap_tok_count + t > overlap_limit:
                        break
                    overlap_sents.insert(0, s)
                    overlap_tok_count += t
                current_sents = overlap_sents
                current_tokens = overlap_tok_count

            current_sents.append(sent)
            current_tokens += sent_tokens

        if current_sents:
            chunks.append(''.join(current_sents))
        return chunks

    @staticmethod
    def is_likely_non_factual(text: str, aggressive: bool = False) -> bool:
        """智能启发式过滤主观、疑问、玩笑等非事实文本"""
        text = text.strip()
        if not text:
            return True

        # 极短文本：检查是否包含数字、专有名词等事实锚点
        if len(text) < 15:
            if re.search(r'\d{2,}', text) or re.search(r'[A-Z][a-z]{2,}', text) or \
               re.search(r'(?:是|在|等于|位于|出生|发明|发现|定理)', text):
                return False
            return True

        for pattern in Config.AUTO_LEARN_BLACKLIST:
            if re.search(pattern, text):
                return True

        if text.endswith('?') or text.endswith('？'):
            return True

        subjective_words = [
            '我觉得', '我认为', '可能', '应该', '也许', '好像', '不知道',
            '大概是', '希望', '喜欢', '讨厌', '感觉', '似乎', '想必',
            '如果', '假设', '倘若', '万一', '假如', '要是'
        ]
        for w in subjective_words:
            if w in text:
                return True

        if re.fullmatch(r'[\W\d_]+', text):
            return True

        if aggressive:
            if len(text) < 30 and not re.search(r'(?:根据|研究|显示|报告|称|宣布|发表|出版)', text):
                return True

        return False

# ----------------------------- 事实核查 -----------------------------
class FactVerifier:
    VERIFY_PROMPT = (
        "你是一名极度严谨的事实核查专家，拥有截至2023年的广泛常识。"
        "请分析以下文本是否包含客观、可验证的事实，并判断其正确性。"
        "注意：\n"
        "- 常识性错误（如'地球是平的'）必须标记为错误。\n"
        "- 缺乏证据的主张、主观看法、未来预测应视为非事实。\n"
        "- 若事实错误，请提供正确版本（如有）。\n"
        "输出严格JSON格式：{\"is_factual\": bool, \"confidence\": float, "
        "\"reason\": \"中文理由\", \"correction\": \"正确事实或空\"}。"
        "不要输出其他内容。"
    )
    _cache: Dict[str, dict] = {}
    _cache_lock = threading.Lock()

    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client

    def verify(self, text: str, source_type: str = "unknown") -> dict:
        # 使用哈希避免长文本键碰撞
        cache_key = hashlib.md5(text.encode()).hexdigest()
        with self._cache_lock:
            if cache_key in self._cache:
                return self._cache[cache_key].copy()

        def _call():
            messages = [
                {"role": "system", "content": self.VERIFY_PROMPT},
                {"role": "user", "content": text}
            ]
            raw = self.llm.chat_completion(messages, temperature=0, max_tokens=200)
            json_match = re.search(r'\{.*\}', raw, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
            else:
                result = json.loads(raw)
            return result

        try:
            result = api_retry(_call)
            is_factual = bool(result.get("is_factual", False))
            confidence = float(result.get("confidence", 0))
            reason = result.get("reason", "")
            correction = result.get("correction", "")

            weight = Config.SOURCE_WEIGHTS.get(source_type, 1.0)
            adjusted_conf = min(confidence * weight, 1.0)

            if len(text.strip()) < 30 and adjusted_conf < Config.SHORT_TEXT_MIN_CONFIDENCE:
                is_factual = False
                reason += " (短文本置信度不足)"

            final_result = {
                "is_factual": is_factual,
                "confidence": adjusted_conf,
                "reason": reason,
                "correction": correction
            }
            with self._cache_lock:
                self._cache[cache_key] = final_result.copy()
            return final_result
        except Exception as e:
            logger.error(f"事实核查异常: {e}", exc_info=True)
            return {"is_factual": False, "confidence": 0, "reason": f"核查异常:{e}", "correction": ""}

class Summarizer:
    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client

    def summarize(self, text: str) -> str:
        def _call():
            messages = [
                {"role": "system", "content": "用一句中文总结以下内容的核心事实，不要添加解释。"},
                {"role": "user", "content": text}
            ]
            return self.llm.chat_completion(messages, temperature=0.2, max_tokens=80)
        try:
            return api_retry(_call)
        except Exception as e:
            logger.error(f"摘要生成失败: {e}")
            return text[:100]

# ----------------------------- 向量数据库 -----------------------------
class VectorStore:
    def __init__(self, db_path=None):
        db_path = db_path or Config.DB_PATH
        self.client = chromadb.PersistentClient(path=db_path)
        self.ef = embedding_functions.OpenAIEmbeddingFunction(
            api_key=Config.OPENAI_API_KEY,
            model_name=Config.EMBEDDING_MODEL
        )
        self.collection = self.client.get_or_create_collection(
            name=Config.COLLECTION_NAME,
            embedding_function=self.ef,
            metadata={"hnsw:space": "cosine"}
        )

    def query(self, text, n_results=1):
        return self.collection.query(
            query_texts=[text],
            n_results=n_results,
            include=["documents", "metadatas", "distances"]
        )

    def add(self, texts, metadatas, ids):
        self.collection.add(documents=texts, metadatas=metadatas, ids=ids)

    def update(self, ids, documents, metadatas):
        self.collection.update(ids=ids, documents=documents, metadatas=metadatas)

    def delete(self, ids):
        self.collection.delete(ids=ids)

    def count(self):
        return self.collection.count()

    def get_all(self, limit=1000, include_deprecated=False) -> List[Dict]:
        all_data = []
        offset = 0
        while True:
            batch = self.collection.get(
                limit=limit,
                offset=offset,
                include=["documents", "metadatas"]
            )
            if not batch['ids']:
                break
            for i in range(len(batch['ids'])):
                meta = batch['metadatas'][i]
                if not include_deprecated and meta.get("deprecated", False):
                    continue
                all_data.append({
                    "id": batch['ids'][i],
                    "content": batch['documents'][i],
                    "metadata": meta
                })
            offset += limit
        return all_data

    def import_data(self, data: List[Dict]):
        texts = []
        metas = []
        ids = []
        for item in data:
            ids.append(item["id"])
            texts.append(item["content"])
            metas.append(item["metadata"])
        self.collection.add(documents=texts, metadatas=metas, ids=ids)
        logger.info(f"导入完成，共 {len(ids)} 条记忆")

# ----------------------------- 记忆管理器 -----------------------------
class MemoryManager:
    def __init__(self, store: VectorStore, summarizer: Summarizer):
        self.store = store
        self.summarizer = summarizer
        self.dialog_memory = deque(maxlen=Config.DIALOG_MAX_TURNS)
        self.dialog_context_header = ""

    def is_duplicate(self, text: str) -> bool:
        try:
            res = self.store.query(text, n_results=1)
            if res["documents"] and res["documents"][0]:
                dist = res["distances"][0][0]
                return (1 - dist) >= Config.SIMILARITY_THRESHOLD
        except Exception as e:
            logger.error(f"去重查询失败: {e}")
        return False

    def find_most_similar(self, text: str) -> Optional[dict]:
        try:
            res = self.store.query(text, n_results=1)
            if res["documents"] and res["documents"][0]:
                doc = res["documents"][0][0]
                meta = res["metadatas"][0][0]
                dist = res["distances"][0][0]
                return {
                    "id": res["ids"][0][0],
                    "document": doc,
                    "metadata": meta,
                    "similarity": 1 - dist
                }
        except Exception as e:
            logger.error(f"相似查询失败: {e}")
        return None

    def add_memory(self, text: str, source: str, confidence: float, reason: str, correction: str = ""):
        if self.is_duplicate(text):
            logger.info(f"跳过重复记忆: {text[:60]}...")
            return

        if correction and len(correction) > 5 and "错" not in correction:
            logger.info(f"事实核查给出纠正，学习纠正版本: {correction}")
            text = correction
            confidence = max(confidence * 0.95, 0.7)
            reason += " (经模型纠正)"

        similar = self.find_most_similar(text)
        if similar and similar["similarity"] >= Config.CONFLICT_SIMILARITY:
            old_conf = similar["metadata"].get("confidence", 0)
            if (confidence > old_conf and
                confidence >= Config.MIN_REPLACE_CONFIDENCE and
                confidence - old_conf > 0.05):
                logger.info(f"知识冲突，新置信度 {confidence:.2f} > 旧 {old_conf:.2f}，将旧知识降权")
                updated_old = {**similar["metadata"], "importance": 0.0, "deprecated": True,
                               "deprecated_by": text[:80]}
                self.store.update(
                    ids=[similar["id"]],
                    documents=[similar["document"]],
                    metadatas=[updated_old]
                )
                summary = self.summarizer.summarize(text)
                new_id = f"{source}_{int(time.time() * 1000)}"
                meta = {
                    "source": source,
                    "confidence": confidence,
                    "reason": reason,
                    "summary": summary,
                    "timestamp": time.time(),
                    "importance": 1.0,
                    "replaces": similar["id"],
                    "conflict": True
                }
                self.store.add(texts=[text], metadatas=[meta], ids=[new_id])
                logger.info(f"记忆固化 (替代旧记忆): {summary}")
                return
            else:
                logger.info(f"冲突但替换条件不满足 (新:{confidence:.2f}, 旧:{old_conf:.2f})，忽略")
                return

        summary = self.summarizer.summarize(text)
        mem_id = f"{source}_{int(time.time() * 1000)}"
        metadata = {
            "source": source,
            "confidence": confidence,
            "reason": reason,
            "summary": summary,
            "timestamp": time.time(),
            "importance": 1.0
        }
        self.store.add(texts=[text], metadatas=[metadata], ids=[mem_id])
        logger.info(f"记忆固化: {summary}")

    def retrieve(self, query: str, top_k: int = 3, filter_deprecated: bool = True) -> List[Dict]:
        """检索并与 ids 对齐，避免遗失记忆 ID"""
        try:
            res = self.store.query(query, n_results=top_k * 2)
            docs = res.get("documents", [[]])
            metas = res.get("metadatas", [[]])
            ids = res.get("ids", [[]])
            if not docs or not docs[0]:
                return []
            results = []
            for i in range(len(docs[0])):
                meta = metas[0][i] if i < len(metas[0]) else {}
                if filter_deprecated and meta.get("deprecated", False):
                    continue
                results.append({
                    "id": ids[0][i] if i < len(ids[0]) else "",
                    "content": docs[0][i],
                    "summary": meta.get("summary", docs[0][i][:100]),
                    "confidence": meta.get("confidence", 0)
                })
                if len(results) >= top_k:
                    break
            return results
        except Exception as e:
            logger.error(f"记忆检索失败: {e}")
            return []

    def find_by_content(self, query: str) -> Optional[dict]:
        similar = self.find_most_similar(query)
        if similar and similar["similarity"] >= Config.CONFLICT_SIMILARITY:
            return similar
        return None

    def delete_memory(self, memory_id: str) -> bool:
        try:
            self.store.delete(ids=[memory_id])
            logger.info(f"已删除记忆: {memory_id}")
            return True
        except Exception as e:
            logger.error(f"删除记忆失败: {e}")
            return False

    def mark_as_false(self, memory_id: str) -> bool:
        try:
            res = self.store.collection.get(ids=[memory_id], include=["metadatas", "documents"])
            if res['ids']:
                old_meta = res['metadatas'][0]
                doc = res['documents'][0]
                new_meta = {**old_meta, "importance": 0.0, "deprecated": True, "marked_false": True}
                self.store.update(ids=[memory_id], documents=[doc], metadatas=[new_meta])
                logger.info(f"已将记忆标记为错误: {memory_id}")
                return True
        except Exception as e:
            logger.error(f"标记错误记忆失败: {e}")
        return False

    def add_dialog_turn(self, role: str, content: str, priority: bool = False):
        self.dialog_memory.append({"role": role, "content": content, "priority": priority})

    def set_dialog_header(self, header: str):
        self.dialog_context_header = header

    def get_dialog_context(self) -> str:
        """以完整问答对为单位保留上下文，并纳入优先级消息的 token 预算"""
        if not self.dialog_memory and not self.dialog_context_header:
            return ""

        enc = get_encoding()
        max_tok = Config.DIALOG_MAX_TOKENS
        header_tok = len(enc.encode(self.dialog_context_header)) if self.dialog_context_header else 0
        remaining_tok = max_tok - header_tok
        if remaining_tok <= 0:
            return self.dialog_context_header

        # 分离优先级消息
        priority_msgs = [t for t in self.dialog_memory if t.get("priority")]
        normal_turns = [t for t in self.dialog_memory if not t.get("priority")]

        # 为 priority 消息按时间保留最近但不超过剩余 token 的一半（留出空间给普通消息）
        priority_limit = remaining_tok // 3
        selected_priority = []
        tok_pri = 0
        for pm in reversed(priority_msgs):
            pm_str = f"{pm['role']}: {pm['content']}"
            tok = len(enc.encode(pm_str))
            if tok_pri + tok > priority_limit:
                break
            selected_priority.insert(0, pm_str)
            tok_pri += tok
        remaining_tok -= tok_pri

        # 普通问答对
        pairs = []
        temp = []
        for turn in normal_turns:
            if turn["role"] == "user":
                if temp and temp[-1]["role"] == "assistant":
                    pairs.append((temp[0], temp[1]) if len(temp) == 2 else (temp[0], None))
                temp = [turn]
            elif turn["role"] == "assistant":
                temp.append(turn)
        if temp:
            pairs.append((temp[0], temp[1] if len(temp) > 1 else None))

        selected_normal = []
        tok_norm = 0
        for user_t, assist_t in reversed(pairs):
            turn_str = f"user: {user_t['content']}"
            if assist_t:
                turn_str += f"\nassistant: {assist_t['content']}"
            tok = len(enc.encode(turn_str))
            if tok_norm + tok > remaining_tok:
                break
            selected_normal.insert(0, turn_str)
            tok_norm += tok

        parts = []
        if self.dialog_context_header:
            parts.append(self.dialog_context_header)
        parts.extend(selected_priority)
        parts.extend(selected_normal)
        return "\n".join(parts)

# ----------------------------- 聊天机器人核心 -----------------------------
class EnhancedChatbot:
    def __init__(self, llm_client=None, store=None, verifier=None, summarizer=None):
        self.llm = llm_client or LLMClient()
        self.store = store or VectorStore()
        self.summarizer = summarizer or Summarizer(self.llm)
        self.verifier = verifier or FactVerifier(self.llm)
        self.memory = MemoryManager(self.store, self.summarizer)
        self.auto_learn = False
        self.learn_queue = deque()
        self._stop_event = threading.Event()
        self._bg_thread = None
        self._start_background_learner()
        self.memory.set_dialog_header("系统：记住，你是基于深度记忆的助手，请尊重已知事实。")
        logger.info("聊天机器人初始化完成")

    def _start_background_learner(self):
        def worker():
            while not self._stop_event.is_set():
                try:
                    if self.learn_queue:
                        text, source = self.learn_queue.popleft()
                        if source.startswith("chat_auto"):
                            if TextProcessor.is_likely_non_factual(text, aggressive=True):
                                logger.debug(f"自动学习激进过滤器跳过: {text[:50]}...")
                                continue
                        self._learn_sync(text, source)
                except Exception as e:
                    logger.error(f"后台学习线程错误: {e}")
                for _ in range(5):
                    if self._stop_event.is_set():
                        break
                    time.sleep(0.1)
        self._bg_thread = threading.Thread(target=worker, daemon=True)
        self._bg_thread.start()

    def shutdown(self):
        logger.info("正在关闭后台学习线程...")
        self._stop_event.set()
        if self._bg_thread and self._bg_thread.is_alive():
            self._bg_thread.join(timeout=3.0)
        if self.learn_queue:
            logger.warning(f"后台队列中仍有 {len(self.learn_queue)} 个未处理任务，已被丢弃")
            self.learn_queue.clear()
        logger.info("后台线程已关闭")

    def _learn_sync(self, text: str, source: str):
        try:
            chunks = TextProcessor.chunk_text_semantically(text)
            logger.info(f"学习: {len(chunks)} 块, 来源:{source}")
            if source.startswith("user_command"):
                src_type = "user_command"
            elif source.startswith("chat_auto"):
                src_type = "chat_auto"
            elif source.startswith("user_correction"):
                src_type = "user_correction"
            else:
                src_type = "file_learn"

            accepted = 0
            for chunk in chunks:
                v = self.verifier.verify(chunk, source_type=src_type)
                if v["is_factual"] and v["confidence"] >= Config.VERIFY_CONFIDENCE_THRESHOLD:
                    self.memory.add_memory(chunk, source, v["confidence"], v["reason"], v.get("correction", ""))
                    accepted += 1
                else:
                    logger.debug(f"丢弃: {v['reason']} | {chunk[:60]}...")
            logger.info(f"学习完成，采纳 {accepted}/{len(chunks)}")
        except Exception as e:
            logger.error(f"同步学习失败: {e}", exc_info=True)

    def learn_from_text(self, text: str, source: str, background: bool = False):
        if len(text) > Config.MAX_LEARN_TEXT_LENGTH:
            logger.warning(f"文本过长 ({len(text)} 字符)，将被截断")
            text = text[:Config.MAX_LEARN_TEXT_LENGTH]
        if background or (self.auto_learn and not source.startswith("user_")):
            self.learn_queue.append((text, source))
        else:
            self._learn_sync(text, source)

    def learn_from_file(self, file_path: str):
        if not os.path.exists(file_path):
            logger.error(f"文件不存在: {file_path}")
            return
        try:
            text = TextProcessor.extract_text_from_file(file_path)
            self.learn_from_text(text, source=file_path)
        except Exception as e:
            logger.error(f"文件学习失败: {e}", exc_info=True)

    def handle_command(self, user_input: str) -> Optional[str]:
        if user_input.startswith("/learn "):
            path = user_input[7:].strip()
            self.learn_from_file(path)
            return f"🤖 已学习文件：{path}"
        elif user_input.startswith("/remember "):
            text = user_input[10:].strip()
            self.learn_from_text(text, source="user_command")
            return "🤖 已记忆其中正确的信息。"
        elif user_input.strip() == "/autolearn on":
            self.auto_learn = True
            return "✅ 自动学习已开启（智能噪声过滤）。"
        elif user_input.strip() == "/autolearn off":
            self.auto_learn = False
            return "⏸️ 自动学习已关闭。"
        elif user_input.strip() == "/stats":
            try:
                cnt = self.store.count()
                return f"📊 记忆库共有 {cnt} 条知识。"
            except:
                return "❌ 无法获取统计信息。"
        elif user_input.startswith("/forget "):
            query = user_input[8:].strip()
            if not query:
                return "❌ 请提供要遗忘的内容关键词。"
            similar = self.memory.find_by_content(query)
            if similar:
                if self.memory.delete_memory(similar["id"]):
                    return f"🗑️ 已删除相关记忆: {similar['document'][:80]}..."
                else:
                    return "❌ 删除失败。"
            else:
                return "🔍 未找到相似记忆。"
        elif user_input.startswith("/flag_false "):
            query = user_input[12:].strip()
            if not query:
                return "❌ 请提供要标记的内容关键词。"
            similar = self.memory.find_by_content(query)
            if similar:
                if self.memory.mark_as_false(similar["id"]):
                    return f"⚠️ 已将记忆标记为错误: {similar['document'][:80]}..."
                else:
                    return "❌ 标记失败。"
            else:
                return "🔍 未找到相似记忆。"
        elif user_input.startswith("/correct "):
            parts = user_input[9:].split("|", 1)
            if len(parts) != 2 or not parts[0].strip() or not parts[1].strip():
                return "❌ 格式：/correct 错误关键词 | 正确事实"
            wrong_part, correct_part = parts[0].strip(), parts[1].strip()
            similar = self.memory.find_by_content(wrong_part)
            if similar:
                self.memory.mark_as_false(similar["id"])
                self.learn_from_text(correct_part, source="user_correction")
                return f"✅ 已用正确信息替换相关记忆：{correct_part[:80]}..."
            else:
                return "🔍 未找到可修正的记忆，使用 /remember 直接记录正确信息。"
        elif user_input.strip() == "/export":
            try:
                all_memories = self.store.get_all(include_deprecated=False)
                if not all_memories:
                    return "📤 记忆库为空或全部已废弃。"
                export_data = []
                for mem in all_memories:
                    export_data.append({
                        "id": mem["id"],
                        "summary": mem["metadata"].get("summary", ""),
                        "confidence": mem["metadata"].get("confidence", 0),
                        "source": mem["metadata"].get("source", ""),
                        "timestamp": mem["metadata"].get("timestamp", 0),
                        "content": mem["content"]
                    })
                export_file = Config.EXPORT_FILENAME.format(timestamp=int(time.time()))
                with open(export_file, 'w', encoding='utf-8') as f:
                    json.dump(export_data, f, ensure_ascii=False, indent=2)
                return f"📤 记忆已导出至 {export_file}，共 {len(export_data)} 条。"
            except Exception as e:
                logger.error(f"导出失败: {e}")
                return "❌ 导出失败，请查看日志。"
        elif user_input.startswith("/import "):
            file_path = user_input[8:].strip()
            if not os.path.exists(file_path):
                return "❌ 导入文件不存在。"
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                if not isinstance(data, list):
                    return "❌ 文件格式错误，需要JSON数组。"
                self.store.import_data(data)
                return f"📥 成功导入 {len(data)} 条记忆。"
            except Exception as e:
                logger.error(f"导入失败: {e}")
                return "❌ 导入失败，请查看日志。"
        return None

    def chat(self, user_input: str) -> str:
        cmd_resp = self.handle_command(user_input)
        if cmd_resp is not None:
            return cmd_resp

        if self.auto_learn and len(user_input.strip()) > 15:
            self.learn_from_text(user_input, source="chat_auto", background=True)

        try:
            memories = self.memory.retrieve(user_input, top_k=3)
        except Exception as e:
            logger.error(f"检索失败: {e}")
            memories = []

        long_term = "已知相关事实：\n" + "\n".join(
            f"- {m['summary']} (信心:{m['confidence']:.2f})" for m in memories
        ) if memories else "（暂无相关知识）"

        dialog_context = self.memory.get_dialog_context()
        system_msg = ("你是拥有深度记忆的AI助手。回答时优先基于已知事实，并用友好中文交流。"
                      "如果某个事实已被标记为错误或过时，请勿使用，并提醒用户。")
        user_prompt = f"{long_term}\n\n对话历史：\n{dialog_context}\n\n用户新问题：{user_input}"

        self.memory.add_dialog_turn("user", user_input)

        def _get_reply():
            return self.llm.chat_completion(
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.5,
                max_tokens=600
            )

        try:
            reply = api_retry(_get_reply)
        except Exception as e:
            logger.error(f"生成回复失败: {e}", exc_info=True)
            reply = "⚠️ 服务暂时不可用，请稍后再试。"

        self.memory.add_dialog_turn("assistant", reply)
        return reply

# ----------------------------- 启动入口 -----------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="深度记忆聊天机器人（生产级完备版）")
    parser.add_argument("--verbose", action="store_true", help="详细日志输出")
    parser.add_argument("--test", action="store_true", help="运行自检")
    args = parser.parse_args()

    if not args.verbose:
        logger.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logger.level)
    formatter = logging.Formatter("%(levelname)s - %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    try:
        Config.validate()
    except Exception as e:
        logger.error(str(e))
        exit(1)

    if args.test:
        print("🔧 运行增强自检...")
        try:
            enc = get_encoding()
            test_text = "这是一个很长的中文句子。" * 100
            chunks = TextProcessor.chunk_text_semantically(test_text)
            assert all(len(enc.encode(c)) <= Config.CHUNK_MAX_TOKENS for c in chunks), "分块超限"
            print(f"✅ 分块测试通过 ({len(chunks)} 块)")

            sents = TextProcessor.split_sentences("你好。今天天气不错！\n我很好。")
            assert len(sents) == 4
            print("✅ 分句测试通过")

            assert TextProcessor.is_likely_non_factual("我觉得很好", aggressive=True)
            assert not TextProcessor.is_likely_non_factual("地球是圆的")
            assert not TextProcessor.is_likely_non_factual("2023年GDP增长5%", aggressive=True)
            print("✅ 智能噪声过滤测试通过")

            bot = EnhancedChatbot()
            bot.memory.add_dialog_turn("user", "你好")
            bot.memory.add_dialog_turn("assistant", "你好！")
            bot.memory.add_dialog_turn("user", "重要提醒：请牢记安全第一", priority=True)
            ctx = bot.memory.get_dialog_context()
            assert "重要提醒" in ctx and "你好" in ctx
            print("✅ 对话上下文与优先级测试通过")

            assert bot.store.count() == 0
            print("✅ 向量库连接正常")
            bot.shutdown()   # 自检完毕释放资源
        except Exception as e:
            logger.error(f"自检失败: {e}", exc_info=True)
        exit(0)

    bot = EnhancedChatbot()
    print("🧠 深度记忆聊天机器人已启动（生产级完备版）")
    print("  /learn <路径>          - 从文件学习")
    print("  /remember <内容>       - 直接记忆")
    print("  /autolearn on/off      - 对话自动学习（智能过滤）")
    print("  /forget <关键词>       - 删除匹配的记忆")
    print("  /flag_false <关键词>   - 标记记忆为错误")
    print("  /correct <错误>|<正确>  - 修正记忆并添加正确版")
    print("  /stats                 - 查看记忆数量")
    print("  /export                - 导出记忆为JSON")
    print("  /import <路径>         - 从JSON导入记忆")
    print("  exit/quit              - 退出\n")
    try:
        while True:
            try:
                user = input("👤 你：")
            except (EOFError, KeyboardInterrupt):
                print("\n🤖 再见！")
                break
            if user.lower() in ['exit', 'quit']:
                print("🤖 再见！")
                break
            reply = bot.chat(user)
            print(f"🤖 AI：{reply}\n")
    finally:
        bot.shutdown()

# ============================================================
# Author: ciain
# Date: 2026-06-05 21:21:28
# ============================================================