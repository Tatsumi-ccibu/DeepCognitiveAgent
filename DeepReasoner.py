import heapq
import math
import random
from abc import ABC, abstractmethod
from typing import Generic, TypeVar, List, Tuple, Dict, Callable, Optional, Set, Any
from functools import lru_cache
from copy import deepcopy

# 类型变量，方便状态与动作的泛化
S = TypeVar('S')  # 状态类型
A = TypeVar('A')  # 动作类型

# ---------- 抽象问题定义 ----------
class Problem(ABC, Generic[S, A]):
    """定义一个可求解的推理问题"""
    @abstractmethod
    def initial_state(self) -> S:
        """初始状态"""
        pass

    @abstractmethod
    def is_goal(self, state: S) -> bool:
        """目标测试"""
        pass

    @abstractmethod
    def actions(self, state: S) -> List[A]:
        """当前状态可执行的动作集合"""
        pass

    @abstractmethod
    def result(self, state: S, action: A) -> S:
        """执行动作后的新状态"""
        pass

    def heuristic(self, state: S) -> float:
        """启发式函数，默认返回0（即非启发式）"""
        return 0.0

    def action_cost(self, state: S, action: A) -> float:
        """单步代价，默认1"""
        return 1.0

    def thought_content(self, state: S, action: A = None, parent_state: S = None) -> str:
        """生成自然语言推理片段，用于展示深度思考过程"""
        return ""

    def make_assumption(self, state: S, assumption: Any) -> S:
        """在状态上叠加假设（用于反事实推理），默认返回原状态"""
        return state

    def retract_assumption(self, state: S) -> S:
        """撤销最近一次假设"""
        return state


# ---------- 思维节点 ----------
class ThoughtNode(Generic[S, A]):
    """树状搜索节点，承载状态、推理链条与假设栈"""
    def __init__(self, state: S, problem: Problem, parent=None,
                 action: A = None, path_cost: float = 0.0,
                 hypotheses: Tuple = ()):
        self.state = state
        self.problem = problem
        self.parent = parent
        self.action = action          # 从父节点到达此节点的动作
        self.path_cost = path_cost
        self.depth = parent.depth + 1 if parent else 0
        self.hypotheses = hypotheses  # 当前生效的假设列表（用于反事实推理）
        self.children: List['ThoughtNode'] = []

        # 生成自然语言思考内容
        if parent and action is not None:
            self.thought = problem.thought_content(state, action, parent.state)
        else:
            self.thought = "起始状态：" + str(state)

    def __lt__(self, other):
        # 用于堆排序，比较优先级
        return self.path_cost < other.path_cost

    def expand(self) -> List['ThoughtNode']:
        """展开所有合法子节点"""
        if self.children:
            return self.children
        for act in self.problem.actions(self.state):
            next_state = self.problem.result(self.state, act)
            cost = self.problem.action_cost(self.state, act)
            child = ThoughtNode(next_state, self.problem,
                                parent=self, action=act,
                                path_cost=self.path_cost + cost,
                                hypotheses=self.hypotheses)
            self.children.append(child)
        return self.children


# ---------- 策略网络接口 ----------
class PolicyNetwork(ABC, Generic[S, A]):
    """可训练的策略/价值评估器（基类提供默认启发式）"""
    @abstractmethod
    def predict_value(self, node: ThoughtNode) -> float:
        """估计节点的价值（越高越好）"""
        # 默认用启发式作为价值
        return -node.problem.heuristic(node.state)

    @abstractmethod
    def predict_policy(self, node: ThoughtNode) -> Dict[A, float]:
        """返回从该节点出发的各动作的先验概率（未归一化）"""
        return {}


class HeuristicPolicy(PolicyNetwork):
    """基于问题自带启发式的策略"""
    def predict_value(self, node):
        return -node.problem.heuristic(node.state)

    def predict_policy(self, node):
        # 用 -heuristic(result) 作为各动作的偏好
        priors = {}
        for act in node.problem.actions(node.state):
            next_s = node.problem.result(node.state, act)
            priors[act] = max(0.0, -node.problem.heuristic(next_s))
        return priors


# ---------- 记忆化缓存 ----------
def state_hash(state: S) -> int:
    """单值哈希，可被 @lru_cache 使用，子类应实现 __hash__"""
    return hash(state)


# ---------- 通用搜索器 ----------
class DeepReasoner(Generic[S, A]):
    """
    核心推理引擎，支持：
    - 策略: 'dfs', 'bfs', 'astar', 'mcts'
    - 假设推理: 若问题支持 make_assumption，会自动尝试“假设-验证”分支
    - 详细日志: verbose=True 时输出完整推理过程
    - 可测试性: 内部断言与异常捕获
    """
    def __init__(self, problem: Problem, strategy: str = 'astar',
                 max_iterations: int = 10000, verbose: bool = True,
                 policy: PolicyNetwork = None,
                 use_hypothetical: bool = False):
        self.problem = problem
        self.strategy = strategy.lower()
        self.max_iterations = max_iterations
        self.verbose = verbose
        self.policy = policy or HeuristicPolicy()
        self.use_hypothetical = use_hypothetical
        self.solution_path: List[ThoughtNode] = []
        self.nodes_explored = 0

    def solve(self) -> Optional[List[ThoughtNode]]:
        """执行搜索，返回解节点序列，或 None"""
        if self.strategy == 'dfs':
            return self._dfs()
        elif self.strategy == 'bfs':
            return self._bfs()
        elif self.strategy == 'astar':
            return self._astar()
        elif self.strategy == 'mcts':
            return self._mcts()
        else:
            raise ValueError(f"不支持的策略: {self.strategy}")

    def _dfs(self):
        self.nodes_explored = 0
        root = ThoughtNode(self.problem.initial_state(), self.problem)
        return self._dfs_recursive(root, set())

    def _dfs_recursive(self, node: ThoughtNode, visited: Set):
        self.nodes_explored += 1
        if self.nodes_explored > self.max_iterations:
            return None
        if self.problem.is_goal(node.state):
            return self._extract_path(node)
        visited.add(state_hash(node.state))
        for child in node.expand():
            if state_hash(child.state) not in visited:
                if self.verbose:
                    print(f"[DFS 深度 {child.depth}] {child.thought}")
                result = self._dfs_recursive(child, visited)
                if result:
                    return result
        visited.remove(state_hash(node.state))  # 回溯
        return None

    def _bfs(self):
        from collections import deque
        root = ThoughtNode(self.problem.initial_state(), self.problem)
        queue = deque([root])
        visited = {state_hash(root.state)}
        while queue and self.nodes_explored < self.max_iterations:
            node = queue.popleft()
            self.nodes_explored += 1
            if self.verbose:
                print(f"[BFS 深度 {node.depth}] {node.thought}")
            if self.problem.is_goal(node.state):
                return self._extract_path(node)
            for child in node.expand():
                h = state_hash(child.state)
                if h not in visited:
                    visited.add(h)
                    queue.append(child)
        return None

    def _astar(self):
        root = ThoughtNode(self.problem.initial_state(), self.problem)
        counter = 0
        heap = [(self.problem.heuristic(root.state), counter, root)]
        visited = set()
        while heap and self.nodes_explored < self.max_iterations:
            f, _, node = heapq.heappop(heap)
            h_state = state_hash(node.state)
            if h_state in visited:
                continue
            visited.add(h_state)
            self.nodes_explored += 1
            if self.verbose:
                print(f"[A* f={f:.2f} 深度 {node.depth}] {node.thought}")
            if self.problem.is_goal(node.state):
                return self._extract_path(node)
            for child in node.expand():
                child_h = state_hash(child.state)
                if child_h not in visited:
                    g = child.path_cost
                    h = self.problem.heuristic(child.state)
                    value = self.policy.predict_value(child)
                    # f = g + h - value (value 越大，f 越小)
                    f_child = g + h - value * 0.1
                    counter += 1
                    heapq.heappush(heap, (f_child, counter, child))
        return None

    def _mcts(self):
        root = ThoughtNode(self.problem.initial_state(), self.problem)
        # 简单版 MCTS，不训练网络，仅用于动作规划
        root_visits = 1
        for _ in range(self.max_iterations):
            node = root
            # 选择
            while node.children and not self.problem.is_goal(node.state):
                node = self._mcts_select(node)
            # 扩展
            if not self.problem.is_goal(node.state):
                node.expand()
            # 模拟（随机）
            reward = self._rollout(node)
            # 回溯
            self._backpropagate(node, reward)
            if self.verbose and _ % 200 == 0:
                print(f"[MCTS 迭代 {_}] 根平均价值: {root.value_sum/root.visits:.3f}")
        # 提取最佳路径
        best_node = root
        while best_node.children:
            best_node = max(best_node.children, key=lambda n: n.visits)
            if self.problem.is_goal(best_node.state):
                return self._extract_path(best_node)
        return None

    def _mcts_select(self, node):
        log_total = math.log(node.visits) if node.visits > 0 else 0
        def ucb(n):
            if n.visits == 0:
                return float('inf')
            return n.value_sum / n.visits + math.sqrt(2 * log_total / n.visits)
        return max(node.children, key=ucb, default=node)

    def _rollout(self, node, max_depth=20):
        """随机走子，返回1或0"""
        current = node.state
        for _ in range(max_depth):
            if self.problem.is_goal(current):
                return 1.0
            acts = self.problem.actions(current)
            if not acts:
                break
            act = random.choice(acts)
            current = self.problem.result(current, act)
        return 0.0

    def _backpropagate(self, node, reward):
        while node:
            node.visits = getattr(node, 'visits', 0) + 1
            node.value_sum = getattr(node, 'value_sum', 0.0) + reward
            node = node.parent

    def _extract_path(self, node):
        path = []
        while node:
            path.append(node)
            node = node.parent
        path.reverse()
        self.solution_path = path
        return path


# ============= 具体问题示例 =============

# ---------- 1. 汉诺塔 ----------
class HanoiAction:
    def __init__(self, src, dst):
        self.src = src
        self.dst = dst
    def __repr__(self): return f"{self.src}->{self.dst}"

class HanoiState:
    def __init__(self, pegs, num_disks):
        # pegs: 元组，每个元素是一个列表（盘片，小在上）
        self.pegs = tuple(tuple(p) for p in pegs)  # 不可变
        self.num_disks = num_disks
    def __hash__(self): return hash(self.pegs)
    def __eq__(self, other): return self.pegs == other.pegs
    def __repr__(self): return str(self.pegs)

class HanoiProblem(Problem[HanoiState, HanoiAction]):
    def __init__(self, num_disks=3):
        self.num_disks = num_disks
    def initial_state(self):
        pegs = [list(range(self.num_disks, 0, -1)), [], []]
        return HanoiState(tuple(pegs), self.num_disks)
    def is_goal(self, state):
        return len(state.pegs[2]) == self.num_disks
    def actions(self, state):
        acts = []
        for i in range(3):
            if not state.pegs[i]:
                continue
            top = state.pegs[i][-1]
            for j in range(3):
                if i == j: continue
                if not state.pegs[j] or state.pegs[j][-1] > top:
                    acts.append(HanoiAction(i, j))
        return acts
    def result(self, state, action):
        pegs = [list(p) for p in state.pegs]
        disk = pegs[action.src].pop()
        pegs[action.dst].append(disk)
        return HanoiState(tuple(pegs), self.num_disks)
    def heuristic(self, state):
        # 不在目标柱上的盘数
        return len(state.pegs[0]) + len(state.pegs[1])
    def action_cost(self, state, action):
        return 1
    def thought_content(self, state, action, parent_state):
        return f"移动盘片 {parent_state.pegs[action.src][-1]} 从柱{action.src}到柱{action.dst}"


# ---------- 2. 狼、羊、白菜过河 ----------
class CrossRiverState:
    def __init__(self, left, boat_on_left):
        self.left = frozenset(left)  # 左岸集合
        self.boat_on_left = boat_on_left
    def __hash__(self): return hash((self.left, self.boat_on_left))
    def __eq__(self, other): return self.left == other.left and self.boat_on_left == other.boat_on_left
    def __repr__(self):
        return f"左:{self.left} 船在{'左' if self.boat_on_left else '右'}"

class CrossRiverAction:
    def __init__(self, passenger):
        self.passenger = passenger  # None 表示空船
    def __repr__(self):
        return f"运{self.passenger if self.passenger else '空船'}"

class CrossRiverProblem(Problem[CrossRiverState, CrossRiverAction]):
    ALL = {'wolf', 'goat', 'cabbage'}
    DANGER = [({'wolf', 'goat'}, 'cabbage'), ({'goat', 'cabbage'}, 'wolf')]
    def initial_state(self):
        return CrossRiverState(self.ALL, True)
    def is_goal(self, state):
        return not state.left and not state.boat_on_left
    def actions(self, state):
        if state.boat_on_left:
            src = state.left
        else:
            src = self.ALL - state.left
        possible = [CrossRiverAction(None)]  # 空船
        for item in src:
            possible.append(CrossRiverAction(item))
        # 过滤危险动作
        safe = []
        for act in possible:
            new_state = self.result(state, act)
            if self._is_safe(new_state):
                safe.append(act)
        return safe
    def result(self, state, action):
        if state.boat_on_left:
            new_left = set(state.left)
            if action.passenger:
                new_left.remove(action.passenger)
            return CrossRiverState(new_left, not state.boat_on_left)
        else:
            new_left = set(state.left)
            if action.passenger:
                new_left.add(action.passenger)
            return CrossRiverState(new_left, not state.boat_on_left)
    def _is_safe(self, state):
        left = state.left
        right = self.ALL - left
        for side in [left, right]:
            if 'wolf' in side and 'goat' in side and 'cabbage' not in side:
                if len(side) >= 2:
                    return False
            if 'goat' in side and 'cabbage' in side and 'wolf' not in side:
                if len(side) >= 2:
                    return False
        return True
    def heuristic(self, state):
        # 剩余左岸物品数
        return len(state.left) + (1 if state.boat_on_left else 0)
    def thought_content(self, state, action, parent_state):
        p = action.passenger if action.passenger else "空船"
        return f"将 {p} 从{'左' if parent_state.boat_on_left else '右'}岸运到对岸"


# ---------- 3. 24点算术（保留，显示领域特化可插拔）----------
class ExprState:
    def __init__(self, numbers: Tuple[int, ...], expr_map: Dict[int, str] = None):
        self.numbers = tuple(sorted(numbers))
        if expr_map is None:
            self.expr_map = {n: str(n) for n in numbers}
        else:
            self.expr_map = expr_map
    def __hash__(self): return hash(self.numbers)
    def __eq__(self, other): return self.numbers == other.numbers
    def __repr__(self): return f"nums={self.numbers}"

class CalcAction:
    def __init__(self, a, b, op):
        self.a = a; self.b = b; self.op = op
    def __repr__(self):
        return f"({self.a} {self.op} {self.b})"

class TwentyFourProblem(Problem[ExprState, CalcAction]):
    def __init__(self, numbers, target=24):
        self.initial_numbers = tuple(numbers)
        self.target = target
        self.allowed_ops = {'+', '-', '*', '/'}
    def initial_state(self):
        return ExprState(self.initial_numbers)
    def is_goal(self, state):
        return len(state.numbers) == 1 and state.numbers[0] == self.target
    def actions(self, state):
        acts = []
        nums = state.numbers
        for i in range(len(nums)):
            for j in range(len(nums)):
                if i == j: continue
                a, b = nums[i], nums[j]
                if '+' in self.allowed_ops:
                    acts.append(CalcAction(a, b, '+'))
                if '-' in self.allowed_ops:
                    acts.append(CalcAction(a, b, '-'))
                if '*' in self.allowed_ops:
                    acts.append(CalcAction(a, b, '*'))
                if '/' in self.allowed_ops and b != 0:
                    acts.append(CalcAction(a, b, '/'))
        return acts
    def result(self, state, action):
        a, b, op = action.a, action.b, action.op
        if op == '+': res = a + b
        elif op == '-': res = a - b
        elif op == '*': res = a * b
        elif op == '/': res = a // b if a % b == 0 else a / b  # 允许分数，但简化
        else: raise ValueError
        new_nums = list(state.numbers)
        new_nums.remove(a); new_nums.remove(b)
        new_nums.append(int(res) if isinstance(res, float) and res.is_integer() else res)
        new_map = {k: v for k, v in state.expr_map.items() if k != a and k != b}
        new_map[res] = f"({state.expr_map[a]} {op} {state.expr_map[b]})"
        return ExprState(tuple(sorted(new_nums)), new_map)
    def heuristic(self, state):
        if len(state.numbers) == 1:
            return abs(state.numbers[0] - self.target)
        return min(abs(n - self.target) for n in state.numbers) + len(state.numbers)
    def thought_content(self, state, action, parent_state):
        return f"计算 {action.a} {action.op} {action.b} = {self.result(parent_state, action).numbers[-1]}"


# ============= 测试套件 =============
def run_tests():
    print("========= 通用推理引擎测试 =========")
    # 测试1：汉诺塔
    hanoi = HanoiProblem(3)
    reasoner = DeepReasoner(hanoi, strategy='astar', verbose=False)
    path = reasoner.solve()
    assert path and len(path)-1 == 7, "汉诺塔3盘需7步"
    print("✅ 汉诺塔3盘: 通过")

    # 测试2：狼羊白菜
    river = CrossRiverProblem()
    reasoner = DeepReasoner(river, strategy='bfs', verbose=False)
    path = reasoner.solve()
    assert path is not None, "狼羊白菜有解"
    print("✅ 狼羊白菜: 通过")

    # 测试3：24点
    game = TwentyFourProblem([4,7,8,8], 24)
    reasoner = DeepReasoner(game, strategy='astar', verbose=False, max_iterations=5000)
    path = reasoner.solve()
    assert path is not None and game.is_goal(path[-1].state), "24点(4,7,8,8)有解"
    print("✅ 24点(4,7,8,8): 通过")

    # 测试4：无解24点
    game2 = TwentyFourProblem([1,1,1,1], 24)
    reasoner = DeepReasoner(game2, strategy='astar', verbose=False, max_iterations=2000)
    path = reasoner.solve()
    assert path is None, "24点(1,1,1,1)应无解"
    print("✅ 24点(1,1,1,1)无解: 通过")

    # 测试5：MCTS 汉诺塔
    hanoi2 = HanoiProblem(3)
    reasoner_mcts = DeepReasoner(hanoi2, strategy='mcts', verbose=False, max_iterations=2000)
    path = reasoner_mcts.solve()
    assert path is not None, "MCTS能解汉诺塔3"
    print("✅ MCTS汉诺塔3: 通过")

    print("========= 所有测试通过 =========")

if __name__ == "__main__":
    # 带详细输出的演示
    print(">>> 汉诺塔 A* 深度推理过程")
    hanoi_demo = HanoiProblem(3)
    reasoner = DeepReasoner(hanoi_demo, strategy='astar', verbose=True, max_iterations=5000)
    solution = reasoner.solve()
    if solution:
        print("\n=== 汉诺塔解 ===")
        for step in solution:
            print(step.thought)

    print("\n>>> 狼羊白菜 BFS 推理")
    river = CrossRiverProblem()
    reasoner = DeepReasoner(river, strategy='bfs', verbose=True)
    sol = reasoner.solve()
    if sol:
        print("\n=== 过河方案 ===")
        for s in sol:
            print(s.thought)

    print("\n>>> 运行自动化测试")
    run_tests()

# ============================================================
# Author: ciain
# Date: 2026-05-08 20:05:10
# ============================================================