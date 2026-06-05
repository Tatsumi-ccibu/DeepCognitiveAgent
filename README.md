# AI 学习与推理项目集

本项目包含两个子项目模块，涵盖 **AI 联网搜索 / 记忆对话** 和 **自主推理 / 自我反思** 两大方向，共 5 个 Python 源文件。

---

## 目录结构

```
├── README.md
├── 爬虫和对话学习/
│   ├── MemoryChatbot.py          # 深度记忆聊天机器人
│   └── WebSearchCrawler.py       # 深度优化 AI 联网搜索爬虫
└── 自主学习和自我反思/
    ├── DeepReasoner.py           # 通用推理引擎（DFS/BFS/A*/MCTS）
    ├── SelfLearningNet.py        # 纯 Python 多层神经网络 + MC Dropout
    └── ReflectiveMind.py         # 具备自我反思与递归推理的心智模型
```

---

## 一、爬虫和对话学习

### 1. MemoryChatbot.py — 深度记忆聊天机器人

[MemoryChatbot.py](爬虫和对话学习/MemoryChatbot.py) 是一个基于 **ChromaDB 向量数据库** 和 **OpenAI API** 的生产级聊天机器人系统。

**核心功能：**

| 模块 | 说明 |
|------|------|
| `Config` | 统一配置管理，支持 `.env` 环境变量覆盖 |
| `LLMClient` | 封装 OpenAI ChatCompletion 调用 |
| `TextProcessor` | 文本分句、语义分块、文件解析（txt/md/pdf/docx）、非事实文本过滤 |
| `FactVerifier` | 调用 LLM 进行事实核查，含缓存与源可信度权重调整 |
| `Summarizer` | 使用 LLM 生成中文单句摘要 |
| `VectorStore` | 基于 ChromaDB 的持久化向量存储，支持增删改查与导入导出 |
| `MemoryManager` | 记忆管理核心：去重、冲突检测与替换、对话上下文管理、优先级消息 |
| `EnhancedChatbot` | 主控制器：后台学习线程、命令系统、记忆检索增强对话 |

**支持的命令：**

| 命令 | 功能 |
|------|------|
| `/learn <路径>` | 从文件学习知识 |
| `/remember <内容>` | 直接记忆文本 |
| `/autolearn on/off` | 对话自动学习开关 |
| `/forget <关键词>` | 删除匹配记忆 |
| `/flag_false <关键词>` | 标记记忆为错误 |
| `/correct <错误>\|<正确>` | 修正记忆 |
| `/stats` | 查看记忆库统计 |
| `/export` | 导出记忆为 JSON |
| `/import <路径>` | 从 JSON 导入记忆 |

**依赖：** `chromadb`, `openai`, `tiktoken`, `python-dotenv`（可选：`pypdf`/`PyPDF2`, `python-docx`）

---

### 2. WebSearchCrawler.py — 深度优化 AI 联网搜索爬虫

[WebSearchCrawler.py](爬虫和对话学习/WebSearchCrawler.py) 是一个多引擎融合的异步搜索与内容抓取系统。

**核心功能：**

| 模块 | 说明 |
|------|------|
| `Config` | API Key 配置、并发/速率限制、权威域名白名单 |
| `SimHash` | 64位 SimHash 指纹计算与海明距离去重 |
| `AsyncFetcher` | 基于 aiohttp 的异步抓取器，支持速率限制、信号量控制、自动重试 |
| `DuckDuckGoEngine` | DuckDuckGo 搜索引擎（无需 API Key） |
| `BingSearchEngine` | Bing Web Search API 集成 |
| `BraveSearchEngine` | Brave Search API 集成 |
| `WikipediaEngine` | Wikipedia API 搜索 |
| `IntentClassifier` | 基于正则的查询意图识别（新闻/评测/事实/学术） |
| `ContentExtractor` | trafilatura 正文提取 + BeautifulSoup fallback |
| `DeepSearchCrawler` | 主检索流程：多引擎并发搜索 → 异步内容抓取 → SimHash 去重 → 语义排序（SentenceTransformer）→ 权威性加权 |

**检索流程：**
1. 多搜索引擎并发查询（Bing → Brave → DuckDuckGo → Wikipedia 故障转移）
2. 异步抓取页面内容（Playwright fallback 处理 JS 渲染页面）
3. SimHash 去重（海明距离 ≤ 3）
4. 语义相似度 + 权威性加权排序（0.7:0.3）
5. 输出结构化结果（含引用链接）

**依赖：** `aiohttp`, `requests`, `beautifulsoup4`, `trafilatura`, `sentence-transformers`, `python-dotenv`（可选：`duckduckgo-search`, `playwright`, `openai`）

---

## 二、自主学习和自我反思

### 3. DeepReasoner.py — 通用推理引擎

[DeepReasoner.py](自主学习和自我反思/DeepReasoner.py) 实现了一个基于抽象问题定义的通用搜索推理框架，支持多种搜索策略和自然语言推理过程展示。

**核心类：**

| 类 | 说明 |
|------|------|
| `Problem<S, A>` | 问题抽象基类：定义初始状态、目标测试、动作集、状态转移、启发式函数 |
| `ThoughtNode<S, A>` | 树状搜索节点：承载状态、推理链条、假设栈 |
| `PolicyNetwork<S, A>` | 策略网络接口：价值预测 + 动作先验概率 |
| `DeepReasoner<S, A>` | 核心推理引擎：支持 DFS / BFS / A* / MCTS 四种策略 |

**内置示例问题：**

| 问题类 | 说明 |
|------|------|
| `HanoiProblem` | 汉诺塔（3盘以上），含启发式函数 |
| `CrossRiverProblem` | 狼、羊、白菜过河问题，自动过滤危险动作 |
| `TwentyFourProblem` | 24点算术游戏，支持加减乘除 |

**搜索策略对比：**

| 策略 | 适用场景 |
|------|------|
| DFS | 深度优先，适合解空间不深的问题 |
| BFS | 广度优先，保证最短路径 |
| A* | 启发式搜索，结合 g(n)+h(n)，效率高 |
| MCTS | 蒙特卡洛树搜索，适合大规模博弈 |

**依赖：** 仅 Python 标准库

---

### 4. SelfLearningNet.py — 纯 Python 多层神经网络

[SelfLearningNet.py](自主学习和自我反思/SelfLearningNet.py) 完全基于 Python 标准库实现了多层前馈神经网络，不依赖 NumPy/PyTorch 等第三方库。

**核心功能：**

| 功能 | 说明 |
|------|------|
| 激活函数 | ReLU、Sigmoid、Tanh（含导数），预留 GELU 接口 |
| 矩阵运算 | 纯 Python 实现 matmul、transpose、broadcast_add、elementwise_apply |
| 权重初始化 | He 初始化（ReLU）/ Xavier 初始化（Sigmoid/Tanh） |
| 前向传播 | 批处理 + Dropout 正则化 |
| 反向传播 | MSE 损失 + L2 正则化 + Dropout 梯度缩放 |
| 训练循环 | 批量训练、学习率衰减、随机打乱 |
| MC Dropout | 递归采样多次带 Dropout 的前向传播取平均，模拟不确定性 |
| 在线学习 | 单样本增量更新权重 |

**演示：** 以 XOR 问题为例，训练一个 2→8→8→1 的深层网络，展示普通预测 vs MC Dropout 深度思考预测的差异。

**依赖：** 仅 Python 标准库（`math`, `random`, `sys`）

---

### 5. ReflectiveMind.py — 自我反思心智模型

[ReflectiveMind.py](自主学习和自我反思/ReflectiveMind.py) 实现了一个具备自我反思、数值推理与递归思考能力的心智模型。

**核心能力：**

| 能力 | 说明 |
|------|------|
| 知识学习 | 存储三元组 (主体, 谓词, 客体)，数值自动转为 `=` 关系 |
| 自然语言解析 | 支持 "X 比 Y 大"、"X 大于 Y"、"X 等于 Y" 等中文句式 |
| 直接检索 | 从记忆中查确切事实，返回信心度 1.0 |
| 数值推理 | 自动提取数值进行 >、<、=、>=、<=、!= 比较 |
| 传递推理 | 支持大于/小于/等于的传递链推理（如 A>B, B>C → A>C） |
| 反向矛盾检测 | 自动检测反向关系以排除矛盾（如已知 A>B，则 C>A 不可能） |
| 联想记忆 | 搜索相关事实辅助推理 |
| 自我反思 | 审查推理文本中的不确定词、循环推理，动态调整信心度 |
| 元认知复盘 | 输出完整思维过程记录与可靠性评估 |
| LRU 缓存 | 推理结果缓存，避免重复计算 |

**推理流程图：**
```
问题 → 自然语言解析 → 直接检索 → 数值推理 → 传递推理 → 反向矛盾检测 → 联想反思 → 自我反思 → 元认知复盘
```

**依赖：** 仅 Python 标准库（`random`, `typing`）

---

## 快速开始

### 对话 & 搜索模块

```bash
cd 爬虫和对话学习

# 安装依赖
pip install chromadb openai tiktoken python-dotenv aiohttp requests beautifulsoup4 trafilatura sentence-transformers

# 设置环境变量
export OPENAI_API_KEY="your-key"

# 启动聊天机器人
python MemoryChatbot.py

# 运行搜索爬虫
python WebSearchCrawler.py
```

### 推理 & 反思模块

```bash
cd 自主学习和自我反思

# 无需安装额外依赖，直接运行

# 推理引擎演示（含自动化测试）
python DeepReasoner.py

# 神经网络训练演示（python SelfLearningNet.py test 仅运行测试）
python SelfLearningNet.py

# 心智模型演示（含自动化测试）
python ReflectiveMind.py
```

---

## 技术特点总结

| 项目 | 关键技术 | 亮点 |
|------|------|------|
| MemoryChatbot | ChromaDB + OpenAI | 事实核查、冲突检测、知识降权、后台异步学习 |
| WebSearchCrawler | aiohttp + SimHash | 多引擎故障转移、语义排序、权威性加权、Playwright fallback |
| DeepReasoner | 泛型 + 策略模式 | DFS/BFS/A*/MCTS 四合一、假设推理预留、自然语言思考链 |
| SelfLearningNet | 纯 Python 矩阵运算 | MC Dropout 深度思考、在线学习、零依赖 |
| ReflectiveMind | 递归推理 + 反思 | 传递推理、矛盾检测、元认知复盘、LRU 缓存 |
