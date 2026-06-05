import random
from typing import Optional, Tuple, Union, Any

class ReflectiveMind:
    """具备自我反思、数值推理与递归思考能力的心智模型（可测试/可调试版）"""

    def __init__(self, name: str = "深思者", seed: int = None, debug: bool = False,
                 max_cache_size: int = 1000, min_confidence: float = 0.7):
        self.name = name
        self.memory = {}          # 记忆库：三元组 (主体, 谓词, 客体) -> True
        self.thought_log = []     # 思考过程记录
        self.min_confidence = min_confidence  # 不确定性阈值
        self.debug = debug        # 是否输出详细思考过程
        self.rng = random.Random(seed)  # 可复现随机
        self.cache = {}           # 推理缓存 {(query, depth): (result, confidence)}
        self.max_cache_size = max_cache_size  # 缓存条目上限

    def set_debug(self, flag: bool):
        """动态开关调试输出"""
        self.debug = flag

    def learn(self, subject: str, predicate: str, obj: Any):
        """
        学习一个事实。
        如果客体是数字，自动转为特殊谓词 '=' (数值型关系) 存储，使数值推理可用。
        """
        if isinstance(obj, (int, float)):
            # 数值统一存储为 (subject, '=', obj)
            self.memory[(subject, '=', obj)] = True
            if self.debug:
                print(f"📚 {self.name}学会了: {subject} = {obj}")
        else:
            self.memory[(subject, predicate, obj)] = True
            if self.debug:
                print(f"📚 {self.name}学会了: {subject} {predicate} {obj}")

    def _log(self, message: str, indent: int = 0):
        """记录思考片段，debug 模式下打印"""
        prefix = "  " * indent
        self.thought_log.append(f"{prefix}{message}")
        if self.debug:
            print(f"{prefix}💭 {message}")

    def ponder(self, question: str) -> Tuple[Optional[str], float]:
        """
        对外思考接口：解析自然语言问题，启动递归思考，最后复盘。
        返回 (结论字符串 或 None, 信心值 0~1)
        """
        self.thought_log.clear()
        if self.debug:
            print(f"\n{'='*40}")
            print(f"❓ 问题: {question}")
            print(f"{'='*40}")

        query = self._parse_question(question)
        if query is None:
            if self.debug:
                print("🤷 无法理解问题。")
            return None, 0.0

        result, confidence = self._deep_think(query, depth=4, intensity=1.0)
        self._meta_reflect(question, result, confidence)
        return result, confidence

    def _parse_question(self, question: str) -> Optional[Tuple[str, str, str]]:
        """
        自然语言解析器（扩展版）
        支持格式：
          - "X 比 Y 大/小"
          - "X 大于/小于 Y"
          - "X 等于 Y" 或 "X 是 Y"
        返回 (主体, 关系符, 客体) 或 None
        TODO: 可扩展"X 的值是多少"等查询
        """
        q = question.replace("吗", "").replace("？", "").replace("?", "").strip()

        # "X 比 Y 大" / "X 比 Y 小"
        if "比" in q:
            parts = q.split()
            if len(parts) >= 3 and parts[1] == "比":
                if "大" in parts[-1]:
                    return (parts[0], ">", parts[2])
                elif "小" in parts[-1]:
                    return (parts[0], "<", parts[2])

        # "X 大于 Y" / "X 小于 Y"
        for keyword, rel in [("大于", ">"), ("小于", "<")]:
            if keyword in q:
                parts = q.replace(keyword, " ").split()
                if len(parts) >= 2:
                    return (parts[0], rel, parts[1])

        # "X 等于 Y" / "X 是 Y"
        for keyword in ["等于", "是"]:
            if keyword in q:
                parts = q.replace(keyword, " ").split()
                if len(parts) >= 2:
                    return (parts[0], "=", parts[1])

        # TODO: 未来可增加 "X 不等于 Y" 等
        return None

    def _get_numeric_value(self, entity: str) -> Optional[Union[int, float]]:
        """从记忆中提取实体对应的数值（如果存在 (entity, '=', number) 记录）"""
        for (sub, pred, obj), flag in self.memory.items():
            if flag and sub == entity and pred == '=' and isinstance(obj, (int, float)):
                return obj
        return None

    def _manage_cache(self):
        """LRU 清理：缓存条目超过上限时删除最旧的一半"""
        if len(self.cache) <= self.max_cache_size:
            return
        remove_count = len(self.cache) // 2
        for _ in range(remove_count):
            self.cache.popitem(last=False)  # Python 3.7+ 保持插入顺序

    def _deep_think(self, query: Tuple[str, str, str], depth: int,
                    intensity: float) -> Tuple[Optional[str], float]:
        """
        核心递归推理函数
        query: (主体, 关系, 客体)
        depth: 剩余递归深度
        intensity: 随机联想概率调节因子
        返回 (解释字符串, 信心值)
        """
        if depth <= 0:
            self._log(f"思考深度耗尽，停止探索 {query}", 4 - depth)
            return None, 0.0

        subject, relation, obj = query
        self._log(f"思考: {subject} {relation} {obj} ? (深度{depth})", 4 - depth)

        # ---------- 缓存检查 ----------
        cache_key = (query, depth)
        if cache_key in self.cache:
            self._log("缓存命中", 4 - depth + 1)
            return self.cache[cache_key]

        # ---------- 1. 直接知识检索 ----------
        if query in self.memory and self.memory[query]:
            self._log(f"记忆中有确切事实: {subject} {relation} {obj}", 4 - depth + 1)
            result = (f"{subject} {relation} {obj}", 1.0)
            self.cache[cache_key] = result
            self._manage_cache()
            return result

        # ---------- 2. 数值推理 ----------
        if relation in (">", "<", ">=", "<=", "=", "!="):
            val_sub = self._get_numeric_value(subject)
            val_obj = self._get_numeric_value(obj)
            if val_sub is not None and val_obj is not None:
                # 进行数值比较
                ops = {
                    ">": lambda a, b: a > b,
                    "<": lambda a, b: a < b,
                    ">=": lambda a, b: a >= b,
                    "<=": lambda a, b: a <= b,
                    "=": lambda a, b: a == b,
                    "!=": lambda a, b: a != b
                }
                if relation in ops:
                    truth = ops[relation](val_sub, val_obj)
                    explanation = f"{subject} = {val_sub}, {obj} = {val_obj}"
                    if truth:
                        explanation += f"，所以 {subject} {relation} {obj}"
                        confidence = 0.99
                    else:
                        explanation += f"，所以 {subject} {relation} {obj} 不成立"
                        confidence = 0.0
                    self._log(f"数值推理{'成功' if truth else '证伪'}: {explanation}", 4 - depth + 1)
                    # 自我反思微调
                    reflected_conf = self._self_reflect(query, explanation, confidence, depth)
                    final_confidence = min(confidence, reflected_conf)
                    result = (explanation if truth else None, final_confidence)
                    self.cache[cache_key] = result
                    self._manage_cache()
                    return result

        # ---------- 3. 传递推理 ----------
        # 定义哪些关系具有传递性
        transitive_map = {
            ">": (">", ">"),
            "<": ("<", "<"),
            "=": ("=", "="),
            ">=": (">=", ">="),
            "<=": ("<=", "<=")
        }
        if relation in transitive_map:
            rel_left, rel_right = transitive_map[relation]
            self._log(f"传递推理: 寻找中介 X 使 {subject} {rel_left} X 且 X {rel_right} {obj}", 4 - depth + 1)
            # 从记忆中收集可能的中间实体
            candidates = set()
            for (s, p, o), flag in self.memory.items():
                if flag and p == rel_left and s == subject:
                    candidates.add(o)
            best_explanation = None
            best_conf = 0.0
            for medium in candidates:
                if medium == subject or medium == obj:
                    continue
                left_query = (subject, rel_left, medium)
                right_query = (medium, rel_right, obj)
                self._log(f"尝试中介 X = {medium}", 4 - depth + 2)
                left_res, left_conf = self._deep_think(left_query, depth - 1, intensity * 0.8)
                if left_res is not None and left_conf > 0:
                    right_res, right_conf = self._deep_think(right_query, depth - 1, intensity * 0.8)
                    if right_res is not None and right_conf > 0:
                        combined = min(left_conf, right_conf) * 0.9
                        if combined > best_conf:
                            best_conf = combined
                            best_explanation = (f"{subject} {rel_left} {medium} 且 {medium} {rel_right} {obj}"
                                                f"，所以 {subject} {relation} {obj}")
            if best_explanation:
                self._log(f"✨ 推理成功: {best_explanation} (信心{best_conf:.2f})", 4 - depth + 1)
                ref_conf = self._self_reflect(query, best_explanation, best_conf, depth)
                final_conf = min(best_conf, ref_conf)
                result = (best_explanation, final_conf)
                self.cache[cache_key] = result
                self._manage_cache()
                return result

        # ---------- 4. 反向矛盾检测 ----------
        reverse_relations = {
            ">": "<", "<": ">", "=": "!=", "!=": "=",
            ">=": "<", "<=": ">"
        }
        if relation in reverse_relations:
            rev_rel = reverse_relations[relation]
            rev_query = (obj, rev_rel, subject)
            self._log(f"反向检测: 检查 {obj} {rev_rel} {subject} 是否成立？", 4 - depth + 1)
            rev_res, rev_conf = self._deep_think(rev_query, depth - 1, intensity * 0.5)
            if rev_res is not None and rev_conf > 0.7:
                self._log(f"反向成立，因此 {subject} {relation} {obj} 不可能", 4 - depth + 1)
                result = (None, 0.0)
                self.cache[cache_key] = result
                self._manage_cache()
                return result

        # ---------- 5. 联想与模糊反思 ----------
        self._log("暂时没有证据，进入更深层反思...", 4 - depth + 1)
        if intensity > 0.5 and self.rng.random() < intensity:
            related_fact = self._find_related_fact(subject, obj)
            if related_fact:
                self._log(f"联想记忆: {related_fact}", 4 - depth + 2)

        # 最终无法得出结论
        result = (None, 0.0)
        self.cache[cache_key] = result
        self._manage_cache()
        return result

    def _find_related_fact(self, entity_a: str, entity_b: str) -> Optional[str]:
        """寻找与两个实体之一相关的已知事实（排除自身查询）"""
        for (s, p, o), flag in self.memory.items():
            if not flag:
                continue
            # 防止返还自身
            if (s == entity_a and o == entity_b) or (s == entity_b and o == entity_a):
                continue
            if s == entity_a or o == entity_a or s == entity_b or o == entity_b:
                return f"{s} {p} {o}"
        return None

    def _self_reflect(self, query: Tuple[str, str, str], explanation: str,
                      confidence: float, depth: int) -> float:
        """自我反思：审查推理文本，寻找漏洞并调整信心"""
        subject, relation, obj = query
        self._log(f"🔍 自我反思: 审查 '{explanation}' 的可靠性", 5 - depth)

        # TODO: 可用更复杂的 NLP 不确定词库
        if "可能" in explanation or "猜测" in explanation:
            self._log("发现不确定词汇，降低信心", 5 - depth + 1)
            return confidence * 0.8

        # 循环推理检测
        conclusion_str = f"{subject} {relation} {obj}"
        if conclusion_str in explanation and explanation.count(conclusion_str) > 1:
            self._log("检测到循环推理，严重怀疑", 5 - depth + 1)
            return 0.2

        # 数值推理通常可靠，但仍保持适度怀疑
        if confidence > 0.95:
            self._log("高度自信，但还是要怀疑一下：是否有反例？", 5 - depth + 1)
            if self.rng.random() < 0.1:
                self._log("突然想到可能忽视了细节，略微降低信心", 5 - depth + 1)
                return 0.9
        return confidence

    def _meta_reflect(self, question: str, result: Optional[str], confidence: float):
        """元认知复盘，输出思维总结"""
        if not self.debug:
            return
        print("\n📝 思维复盘:")
        if result is None:
            print("  - 未能解答，知识库中似乎缺失关键信息。")
            print("  - 建议：学习更多关于该问题的关系。")
        else:
            print(f"  - 结论 '{result}' 的信心度为 {confidence:.2f}")
            if confidence < self.min_confidence:
                print("  - 反思：这个结论可能不可靠，需要更多证据支撑。")
            else:
                print("  - 反思：推理链条经过了自我审查，相对可靠。")
        max_show = 5
        print("  - 思维过程记录:")
        for line in self.thought_log[:max_show]:
            print(f"    {line}")
        if len(self.thought_log) > max_show:
            print(f"    ... 共 {len(self.thought_log)} 步思考")


# ================= 测试用例（可独立运行） =================
def test_transitive_greater():
    """传递推理：A > B > C => A > C"""
    print("=== 测试1: 传递推理（大于关系）===")
    mind = ReflectiveMind("测试者", seed=42, debug=True)
    mind.learn("A", ">", "B")
    mind.learn("B", ">", "C")
    mind.learn("C", ">", "D")
    result, conf = mind.ponder("A 比 C 大吗?")
    assert result is not None and "A >" in result and "C" in result
    assert conf > 0.7
    print("✅ 测试1通过\n")

def test_reverse_doubt():
    """反向问题自我质疑：已知 A > B > C，问 C > A 应返回None"""
    print("=== 测试2: 反向问题（自我质疑）===")
    mind = ReflectiveMind("测试者", seed=42, debug=True)
    mind.learn("A", ">", "B")
    mind.learn("B", ">", "C")
    result, conf = mind.ponder("C 比 A 大吗?")
    assert result is None
    print("✅ 测试2通过\n")

def test_numeric_comparison():
    """数值推理：X=5, Y=3 => X > Y 成立"""
    print("=== 测试3: 数值推理（直接比较）===")
    mind = ReflectiveMind("测试者", seed=42, debug=True)
    mind.learn("X", "=", 5)
    mind.learn("Y", "=", 3)
    result, conf = mind.ponder("X 比 Y 大吗?")
    assert result is not None and "X = 5" in result and "Y = 3" in result
    assert conf > 0.7
    print("✅ 测试3通过\n")

def test_mixed_symbolic_numeric():
    """混合推理：A > B, B = 7, C = 5 => A > C 成立（需传递+数值）"""
    print("=== 测试4: 符号+数值混合推理 ===")
    mind = ReflectiveMind("测试者", seed=42, debug=True)
    mind.learn("A", ">", "B")
    mind.learn("B", "=", 7)
    mind.learn("C", "=", 5)
    mind.learn("B", ">", "C")  # 也可直接数值比较 B > C
    result, conf = mind.ponder("A 比 C 大吗?")
    assert result is not None and conf > 0.7
    print("✅ 测试4通过\n")

def test_unknown_query():
    """完全未知问题，应返回 None"""
    print("=== 测试5: 完全未知问题 ===")
    mind = ReflectiveMind("测试者", seed=42, debug=False)
    mind.learn("X", ">", "Y")
    result, conf = mind.ponder("F 比 G 大吗?")
    assert result is None
    print("✅ 测试5通过\n")


if __name__ == "__main__":
    # ---------- 手动演示 ----------
    demo = ReflectiveMind("苏格拉底", seed=123, debug=True)
    demo.learn("A", ">", "B")
    demo.learn("B", ">", "C")
    demo.learn("C", ">", "D")
    demo.ponder("A 比 C 大吗?")
    print("\n" + "=" * 40)
    demo.ponder("C 比 A 大吗?")
    print("\n" + "=" * 40)
    demo.ponder("F 比 G 大吗?")
    print("\n" + "=" * 40)

    # 数值推理演示
    demo2 = ReflectiveMind("亚里士多德", seed=456, debug=True)
    demo2.learn("身高A", "=", 180)
    demo2.learn("身高B", "=", 175)
    demo2.ponder("身高A 比 身高B 大吗?")

    # ---------- 自动化测试 ----------
    test_transitive_greater()
    test_reverse_doubt()
    test_numeric_comparison()
    test_mixed_symbolic_numeric()
    test_unknown_query()

# ============================================================
# Author: ciain
# Date: 2026-05-08 21:09:36
# ============================================================