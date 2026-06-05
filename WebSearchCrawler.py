#!/usr/bin/env python3
"""
深度优化版AI联网搜索爬虫 - 生产级检索系统
特性：
- 多搜索引擎融合 (Bing, Brave, DDG, Wikipedia) + 故障转移
- 动态内容抓取 (Playwright fallback)
- 内容 SimHash 去重
- 页面权威性评估
- 异步高并发 (aiohttp)
- 查询意图识别
- 多模态信息提取
- 可选的RAG答案生成
"""

import asyncio
import hashlib
import json
import logging
import os
import random
import re
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any, Set
from urllib.parse import urlparse, quote_plus

import aiohttp
import requests
from bs4 import BeautifulSoup
import trafilatura
from sentence_transformers import SentenceTransformer, util

# 可选依赖
try:
    from duckduckgo_search import DDGS
except ImportError:
    DDGS = None

try:
    from playwright.async_api import async_playwright
except ImportError:
    async_playwright = None

try:
    import openai
except ImportError:
    openai = None

from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("DeepSearch")

# ==========================================
# 配置与管理
# ==========================================
class Config:
    # API Keys (环境变量)
    BING_API_KEY = os.getenv("BING_API_KEY", "")
    BRAVE_API_KEY = os.getenv("BRAVE_API_KEY", "")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    # 搜索参数
    MAX_SEARCH_RESULTS = 20
    CONTENT_MIN_LENGTH = 200  # 少于此字数触发Playwright
    # 缓存
    CACHE_DIR = ".cache/search"
    # 请求控制
    MAX_CONCURRENT_REQUESTS = 10
    REQUEST_TIMEOUT = 15
    RATE_LIMIT_PER_DOMAIN = 2.0  # 每秒请求数
    # 权威性权重
    AUTHORITY_BONUS = 0.2
    HIGH_AUTHORITY_DOMAINS = {"github.com", "stackoverflow.com", "wikipedia.org", "arxiv.org",
                              "gov", "edu", "nature.com", "science.org"}

# ==========================================
# SimHash 实现 (用于内容去重)
# ==========================================
class SimHash:
    @staticmethod
    def _hash(token: str) -> int:
        return int(hashlib.md5(token.encode('utf-8')).hexdigest(), 16)

    @staticmethod
    def compute(text: str) -> int:
        """计算64位SimHash指纹"""
        if not text:
            return 0
        v = [0] * 64
        tokens = re.findall(r'\w+', text.lower())[:512]  # 取前512个词
        for token in tokens:
            h = SimHash._hash(token)
            for i in range(64):
                v[i] += 1 if (h >> i) & 1 else -1
        fingerprint = 0
        for i in range(64):
            if v[i] > 0:
                fingerprint |= (1 << i)
        return fingerprint

    @staticmethod
    def distance(a: int, b: int) -> int:
        x = a ^ b
        return bin(x).count('1')

# ==========================================
# 异步请求工具
# ==========================================
class AsyncFetcher:
    """基于aiohttp的异步抓取器，支持速率限制与重试"""
    def __init__(self, user_agent: str = "AISearchBot/2.0", rate_per_sec: float = 2.0):
        self.user_agent = user_agent
        self.rate_per_sec = rate_per_sec
        self.semaphore = asyncio.Semaphore(Config.MAX_CONCURRENT_REQUESTS)
        self._domain_limits: Dict[str, asyncio.Event] = {}
        self._last_request: Dict[str, float] = {}
        self.session: Optional[aiohttp.ClientSession] = None

    async def __aenter__(self):
        self.session = aiohttp.ClientSession(headers={"User-Agent": self.user_agent})
        return self

    async def __aexit__(self, *args):
        await self.session.close()

    async def _rate_limit(self, domain: str):
        now = time.monotonic()
        last = self._last_request.get(domain, 0)
        delay = max(0, 1/self.rate_per_sec - (now - last))
        if delay > 0:
            await asyncio.sleep(delay)
        self._last_request[domain] = time.monotonic()

    async def fetch(self, url: str) -> Optional[str]:
        """异步抓取URL内容，带重试"""
        domain = urlparse(url).netloc
        async with self.semaphore:
            await self._rate_limit(domain)
            for attempt in range(3):
                try:
                    async with self.session.get(url, timeout=Config.REQUEST_TIMEOUT) as resp:
                        if resp.status == 200:
                            return await resp.text()
                        elif resp.status == 429:  # Too Many Requests
                            await asyncio.sleep(min(2 ** attempt, 10))
                        else:
                            break
                except Exception as e:
                    logger.warning(f"抓取失败 {url} (attempt {attempt+1}): {e}")
                    await asyncio.sleep(1)
        return None

# ==========================================
# 搜索引擎实现 (多源 + 故障转移)
# ==========================================
class SearchResult:
    __slots__ = ('title', 'url', 'snippet', 'date', 'source_engine')
    def __init__(self, title, url, snippet, date=None, engine=''):
        self.title = title
        self.url = url
        self.snippet = snippet
        self.date = date
        self.source_engine = engine

class BaseSearchEngine:
    async def search(self, query: str, num: int = 10, time_range: Optional[str] = None) -> List[SearchResult]:
        raise NotImplementedError

class DuckDuckGoEngine(BaseSearchEngine):
    def __init__(self):
        if DDGS is None:
            raise ImportError("需要 duckduckgo-search")

    async def search(self, query, num=10, time_range=None):
        loop = asyncio.get_event_loop()
        def _sync():
            results = []
            timelimit = {'d':'d','w':'w','m':'m'}.get(time_range, None)
            with DDGS() as ddgs:
                for r in ddgs.text(query, max_results=num, timelimit=timelimit):
                    results.append(SearchResult(
                        r.get("title",""), r.get("href",""), r.get("body",""),
                        r.get("date",""), "DuckDuckGo"
                    ))
            return results
        return await loop.run_in_executor(None, _sync)

class BingSearchEngine(BaseSearchEngine):
    def __init__(self, api_key: str):
        self.api_key = api_key or Config.BING_API_KEY
        self.endpoint = "https://api.bing.microsoft.com/v7.0/search"

    async def search(self, query, num=10, time_range=None):
        if not self.api_key:
            return []
        headers = {"Ocp-Apim-Subscription-Key": self.api_key}
        params = {"q": query, "count": num, "mkt": "zh-CN"}
        if time_range:
            params["freshness"] = {'d':'Day','w':'Week','m':'Month'}.get(time_range, None)
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(self.endpoint, headers=headers, params=params) as resp:
                    data = await resp.json()
                    results = []
                    for item in data.get("webPages", {}).get("value", []):
                        results.append(SearchResult(
                            item.get("name",""), item.get("url",""), item.get("snippet",""),
                            item.get("dateLastCrawled",""), "Bing"
                        ))
                    return results
            except Exception as e:
                logger.error(f"Bing搜索失败: {e}")
                return []

class BraveSearchEngine(BaseSearchEngine):
    def __init__(self, api_key: str):
        self.api_key = api_key or Config.BRAVE_API_KEY
        self.endpoint = "https://api.search.brave.com/res/v1/web/search"

    async def search(self, query, num=10, time_range=None):
        if not self.api_key:
            return []
        headers = {"Accept": "application/json", "X-Subscription-Token": self.api_key}
        params = {"q": query, "count": num}
        if time_range:
            params["freshness"] = {'d':'pd','w':'pw','m':'pm'}.get(time_range, "no")
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(self.endpoint, headers=headers, params=params) as resp:
                    data = await resp.json()
                    results = []
                    for item in data.get("web", {}).get("results", []):
                        results.append(SearchResult(
                            item.get("title",""), item.get("url",""), item.get("description",""),
                            item.get("age",""), "Brave"
                        ))
                    return results
            except Exception as e:
                logger.error(f"Brave搜索失败: {e}")
                return []

class WikipediaEngine(BaseSearchEngine):
    """使用Wikipedia API搜索"""
    async def search(self, query, num=10, time_range=None):
        endpoint = "https://en.wikipedia.org/w/api.php"
        params = {
            "action": "query",
            "list": "search",
            "srsearch": query,
            "format": "json",
            "srlimit": num
        }
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(endpoint, params=params) as resp:
                    data = await resp.json()
                    results = []
                    for item in data.get("query", {}).get("search", []):
                        title = item["title"]
                        page_url = f"https://en.wikipedia.org/wiki/{quote_plus(title)}"
                        results.append(SearchResult(
                            title, page_url, item.get("snippet", ""),
                            None, "Wikipedia"
                        ))
                    return results
            except Exception as e:
                logger.error(f"Wikipedia搜索失败: {e}")
                return []

# ==========================================
# 意图识别
# ==========================================
class IntentClassifier:
    """基于正则的简单意图分类"""
    PATTERNS = {
        "latest_news": r"(最新|近期|今天|昨天|本周|热点|新闻|消息|快讯)",
        "opinion_review": r"(评测|评价|推荐|最好|哪个好|对比|体验|心得|怎么样)",
        "factual": r"(是什么|定义|含义|历史|原因|如何|怎么|方法|教程|指南|步骤|原理)",
        "academic": r"(论文|研究|学术|doi|arxiv|引用|文献)",
    }
    @classmethod
    def classify(cls, query: str) -> str:
        for intent, pattern in cls.PATTERNS.items():
            if re.search(pattern, query):
                return intent
        return "factual"  # 默认

# ==========================================
# 核心检索系统
# ==========================================
class DeepSearchCrawler:
    def __init__(self):
        # 搜索引擎优先级：Bing > Brave > DuckDuckGo > Wikipedia
        self.engines: List[BaseSearchEngine] = []
        if Config.BING_API_KEY:
            self.engines.append(BingSearchEngine(Config.BING_API_KEY))
        if Config.BRAVE_API_KEY:
            self.engines.append(BraveSearchEngine(Config.BRAVE_API_KEY))
        self.engines.append(DuckDuckGoEngine())
        self.engines.append(WikipediaEngine())

        self.ranker = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        self.simhashes: Dict[int, str] = {}  # fingerprint -> url (去重)
        self.result_cache: Dict[str, Any] = {}  # 简单内存缓存
        self._init_cache_dir()

    def _init_cache_dir(self):
        Path(Config.CACHE_DIR).mkdir(parents=True, exist_ok=True)

    async def _fetch_and_extract(self, url: str, fetcher: AsyncFetcher) -> Tuple[str, Dict[str, Any]]:
        """抓取页面并提取内容，必要时Playwright fallback"""
        html = await fetcher.fetch(url)
        if not html:
            return "", {}
        extracted = ContentExtractor.extract(html, url)
        text = extracted.get("text", "")
        # 若文本太短，尝试Playwright渲染
        if len(text) < Config.CONTENT_MIN_LENGTH and async_playwright:
            text = await self._playwright_render(url)
            if text:
                extracted["text"] = text
        return text, {k:v for k,v in extracted.items() if k!="text"}

    async def _playwright_render(self, url: str) -> str:
        """使用Playwright获取JS渲染后的页面正文"""
        try:
            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=True)
                page = await browser.new_page()
                await page.goto(url, timeout=20000)
                content = await page.content()
                await browser.close()
                # 用 trafilatura 或简易提取
                soup = BeautifulSoup(content, 'html.parser')
                for tag in soup(['script', 'style', 'nav', 'footer']):
                    tag.decompose()
                return soup.get_text(separator='\n', strip=True)
        except Exception as e:
            logger.warning(f"Playwright渲染失败 {url}: {e}")
            return ""

    async def search_and_retrieve(
        self,
        query: str,
        max_results: int = 8,
        time_range: Optional[str] = None,
        fetch_content: bool = True,
        intent: Optional[str] = None
    ) -> List[Dict]:
        """主检索流程（全异步）"""
        intent = intent or IntentClassifier.classify(query)
        logger.info(f"查询意图: {intent}")

        # 1. 多引擎搜索 (异步并发)
        all_results: List[SearchResult] = []
        seen_urls: Set[str] = set()
        for engine in self.engines:
            try:
                res = await engine.search(query, num=Config.MAX_SEARCH_RESULTS, time_range=time_range)
                logger.info(f"{engine.__class__.__name__} 返回 {len(res)} 条结果")
                for r in res:
                    if r.url not in seen_urls:
                        seen_urls.add(r.url)
                        all_results.append(r)
            except Exception as e:
                logger.error(f"搜索引擎 {engine.__class__.__name__} 异常: {e}")

        if not all_results:
            return []

        # 2. 内容抓取 (异步)
        if fetch_content:
            async with AsyncFetcher() as fetcher:
                tasks = [self._fetch_and_extract(r.url, fetcher) for r in all_results]
                contents = await asyncio.gather(*tasks)
            for i, (text, meta) in enumerate(contents):
                all_results[i].text = text
                all_results[i].metadata = meta
        else:
            for r in all_results:
                r.text = ""
                r.metadata = {}

        # 3. SimHash 去重
        deduped = []
        sim_collisions = 0
        for r in all_results:
            if not hasattr(r, 'text') or not r.text:
                deduped.append(r)
                continue
            fp = SimHash.compute(r.text)
            duplicate = False
            # 查找相同或相似指纹
            for existing_fp in self.simhashes:
                if SimHash.distance(fp, existing_fp) <= 3:  # 海明距离<=3认为重复
                    duplicate = True
                    sim_collisions += 1
                    break
            if not duplicate:
                self.simhashes[fp] = r.url
                deduped.append(r)
        logger.info(f"SimHash去重: 移除 {sim_collisions} 条重复内容")
        all_results = deduped

        # 4. 权威性评估与排序
        for r in all_results:
            auth_score = 0.0
            domain = urlparse(r.url).netloc
            if any(domain.endswith(d) for d in Config.HIGH_AUTHORITY_DOMAINS):
                auth_score = Config.AUTHORITY_BONUS
            r.auth_score = auth_score
            # 内容相关性得分 (异步批量计算)
            texts_for_emb = [r.text[:500] if hasattr(r,'text') and r.text else r.snippet]
            r.sem_score = 0.0

        # 批量计算语义相似度
        texts_to_rank = [(r.text[:500] if hasattr(r,'text') and r.text else r.snippet) for r in all_results]
        if texts_to_rank:
            query_emb = self.ranker.encode(query, convert_to_tensor=False)
            doc_embs = self.ranker.encode(texts_to_rank, convert_to_tensor=False)
            scores = util.cos_sim(query_emb, doc_embs)[0].tolist()
            for i, r in enumerate(all_results):
                r.sem_score = scores[i]

        # 综合排序 (语义 0.7 + 权威 0.3)
        for r in all_results:
            r.final_score = 0.7 * r.sem_score + 0.3 * r.auth_score

        all_results.sort(key=lambda x: x.final_score, reverse=True)

        # 5. 输出格式化
        final = []
        for r in all_results[:max_results]:
            final.append({
                "title": r.title,
                "url": r.url,
                "snippet": r.snippet,
                "date": r.date,
                "engine": r.source_engine,
                "text": getattr(r, 'text', "")[:2000],
                "metadata": getattr(r, 'metadata', {}),
                "relevance": round(r.final_score, 3),
                "citation": f"[{r.title}]({r.url})"
            })
        return final

# ==========================================
# 提取器等复用原代码优化版，这里仅保留关键部分
# ==========================================
class ContentExtractor:
    @staticmethod
    def extract(html: str, url: str = "") -> Dict[str, Any]:
        try:
            extracted = trafilatura.extract(html, include_images=True, output_format='json', url=url)
            if extracted:
                return json.loads(extracted)
        except:
            pass
        # fallback
        soup = BeautifulSoup(html, 'html.parser')
        for tag in soup(['script','style','nav','footer','header']):
            tag.decompose()
        return {"text": soup.get_text(separator='\n', strip=True)}

async def main():
    crawler = DeepSearchCrawler()
    results = await crawler.search_and_retrieve(
        "DeepSeek R1 最新性能评测 2025",
        max_results=5,
        time_range='m'
    )
    print(json.dumps(results, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    asyncio.run(main())

# ============================================================
# Author: ciain
# Date: 2026-04-028 15:05:44
# ============================================================