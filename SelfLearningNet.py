"""
多层神经网络 with MC Dropout 深度思考
完全基于 Python 标准库实现（不使用 NumPy/PyTorch 等）
特点：
- 深层网络（可配置任意层数）
- 激活函数：ReLU, Sigmoid, Tanh（预留 GELU 接口）
- 批处理训练 + L2 正则 + Dropout
- 递归 MC Dropout 模拟“深度思考”预测
- 在线学习（持续更新）
- 内置测试与 TODO 标记
"""
import math
import random
import sys

# 提高递归深度以支持深度思考（MC Dropout 递归）
sys.setrecursionlimit(10000)


# ========================= 激活函数 =========================
def relu(x):
    """ReLU 激活函数"""
    return max(0.0, x)


def relu_derivative(activated):
    """ReLU 导数（输入为激活后的值）"""
    return 1.0 if activated > 0.0 else 0.0


def sigmoid(x):
    """Sigmoid 激活函数（数值稳定）"""
    # 裁剪防止溢出
    x = max(-50.0, min(50.0, x))
    return 1.0 / (1.0 + math.exp(-x))


def sigmoid_derivative(activated):
    """Sigmoid 导数（输入为激活后的值）"""
    return activated * (1.0 - activated)


def tanh(x):
    """Tanh 激活函数"""
    return math.tanh(x)


def tanh_derivative(activated):
    """Tanh 导数（输入为激活后的值）"""
    return 1.0 - activated * activated


# TODO: 实现 GELU 及其导数
def get_activation(act_type):
    """
    根据类型获取 (激活函数, 导数函数)
    类型: 0 -> ReLU, 1 -> Sigmoid, 2 -> Tanh
    """
    if act_type == 0:
        return relu, relu_derivative
    elif act_type == 1:
        return sigmoid, sigmoid_derivative
    elif act_type == 2:
        return tanh, tanh_derivative
    else:
        raise ValueError(f"不支持的激活类型: {act_type}")


# ========================= 基础矩阵运算 =========================
def matmul(A, B):
    """
    矩阵乘法 A (m×n) × B (n×p) -> C (m×p)
    纯 Python 实现，性能可接受于小规模网络
    """
    m = len(A)
    n = len(A[0])
    p = len(B[0])
    # 校验维度匹配
    assert len(B) == n, f"矩阵维度不匹配: A 的列数 {n} != B 的行数 {len(B)}"
    C = [[0.0] * p for _ in range(m)]
    for i in range(m):
        Ai = A[i]
        Ci = C[i]
        for k in range(n):
            aik = Ai[k]
            if aik == 0.0:  # 小优化：跳过零元素
                continue
            Bk = B[k]
            for j in range(p):
                Ci[j] += aik * Bk[j]
    return C


def transpose(M):
    """矩阵转置"""
    return list(map(list, zip(*M)))


def broadcast_add(X, bias):
    """
    广播加偏置：X (m×n) + bias (n)
    返回新矩阵
    """
    return [[X[i][j] + bias[j] for j in range(len(bias))] for i in range(len(X))]


def elementwise_apply(M, func):
    """对矩阵所有元素应用函数"""
    return [[func(x) for x in row] for row in M]


# ========================= 网络初始化 =========================
def initialize_network(layers, activations):
    """
    初始化全连接网络
    layers: 各层神经元数量列表，例如 [2, 8, 8, 1]
    activations: 每个隐藏层及输出层的激活类型列表，长度 = len(layers)-1
    """
    assert len(layers) >= 2, "至少需要输入层和输出层"
    assert len(activations) == len(layers) - 1, "激活函数数量应与层数-1一致"

    net = {'layers': layers, 'activations': activations}
    for i in range(len(layers) - 1):
        in_dim = layers[i]
        out_dim = layers[i + 1]
        # 权重初始化策略
        # TODO: 支持更多初始化方法（如正交初始化）
        act_type = activations[i]
        if act_type == 0:  # ReLU -> He 初始化
            std = math.sqrt(2.0 / in_dim)
        else:  # 其他 -> Xavier 初始化
            std = math.sqrt(1.0 / in_dim)
        weights = [[random.gauss(0.0, std) for _ in range(out_dim)] for _ in range(in_dim)]
        biases = [0.0] * out_dim
        net[f'w{i}'] = weights
        net[f'b{i}'] = biases
    return net


# ========================= 前向传播（批处理 + Dropout） =========================
def forward(net, X, dropout_rate=0.0):
    """
    X: 输入矩阵，shape (batch_size, input_dim)
    dropout_rate: 隐藏层 Dropout 概率，0 表示不丢弃
    返回 (cache_list, output_matrix)
    """
    cache = []
    A = X
    num_layers = len(net['layers']) - 1
    for i in range(num_layers):
        W = net[f'w{i}']
        b = net[f'b{i}']
        Z = broadcast_add(matmul(A, W), b)
        mask = None

        if i < num_layers - 1:  # 隐藏层
            act_fn, _ = get_activation(net['activations'][i])
            A_next = elementwise_apply(Z, act_fn)
            # 应用 Dropout
            if dropout_rate > 0.0:
                scale = 1.0 / (1.0 - dropout_rate)
                rows, cols = len(A_next), len(A_next[0])
                mask = [[scale if random.random() > dropout_rate else 0.0 for _ in range(cols)] for _ in range(rows)]
                A_next = [[A_next[r][c] * mask[r][c] for c in range(cols)] for r in range(rows)]
        else:  # 输出层（不使用 Dropout）
            act_fn, _ = get_activation(net['activations'][i])
            A_next = elementwise_apply(Z, act_fn)

        cache.append({'z': Z, 'a': A_next, 'mask': mask})
        A = A_next

    return cache, A


# ========================= 反向传播（批处理 + L2 正则） =========================
def backward(net, X, Y, cache, learning_rate, reg_lambda=0.0):
    """
    X, Y: 输入和目标矩阵（batch_size × ...）
    cache: forward 缓存的列表
    learning_rate: 学习率
    reg_lambda: L2 正则化系数
    """
    batch_size = len(X)
    num_layers = len(net['layers']) - 1

    # 输出层误差
    last_cache = cache[-1]
    A_out = last_cache['a']
    # 假设损失为 MSE，无法直接换损失函数
    # TODO: 支持交叉熵等损失
    dA = [[(A_out[i][j] - Y[i][j]) for j in range(len(Y[0]))] for i in range(batch_size)]
    _, der_fn = get_activation(net['activations'][-1])
    dZ = [[dA[i][j] * der_fn(A_out[i][j]) for j in range(len(dA[0]))] for i in range(batch_size)]

    # 更新输出层权重
    A_prev = X if num_layers == 1 else cache[-2]['a']
    dW = matmul(transpose(A_prev), dZ)
    dW = [[x / batch_size for x in row] for row in dW]
    db = [sum(dZ[row][col] for row in range(batch_size)) / batch_size for col in range(len(dZ[0]))]

    W = net[f'w{num_layers - 1}']
    b = net[f'b{num_layers - 1}']
    for r in range(len(W)):
        for c in range(len(W[0])):
            W[r][c] -= learning_rate * dW[r][c] + reg_lambda * W[r][c]  # L2 衰减
    for c in range(len(b)):
        b[c] -= learning_rate * db[c]

    # 反向传播到前面各层
    dA_prev = matmul(dZ, transpose(net[f'w{num_layers - 1}']))
    for l in range(num_layers - 2, -1, -1):
        cur_cache = cache[l]
        A_cur, mask = cur_cache['a'], cur_cache['mask']
        # 应用 Dropout mask 的梯度缩放
        if mask is not None:
            dA_prev = [[dA_prev[i][j] * mask[i][j] for j in range(len(dA_prev[0]))] for i in range(batch_size)]

        _, der_fn = get_activation(net['activations'][l])
        dZ_cur = [[dA_prev[i][j] * der_fn(A_cur[i][j]) for j in range(len(dA_prev[0]))] for i in range(batch_size)]
        A_prev_layer = X if l == 0 else cache[l - 1]['a']
        dW = matmul(transpose(A_prev_layer), dZ_cur)
        dW = [[x / batch_size for x in row] for row in dW]
        db = [sum(dZ_cur[row][col] for row in range(batch_size)) / batch_size for col in range(len(dZ_cur[0]))]

        W = net[f'w{l}']
        b = net[f'b{l}']
        for r in range(len(W)):
            for c in range(len(W[0])):
                W[r][c] -= learning_rate * dW[r][c] + reg_lambda * W[r][c]
        for c in range(len(b)):
            b[c] -= learning_rate * db[c]

        if l > 0:
            dA_prev = matmul(dZ_cur, transpose(net[f'w{l}']))


# ========================= 训练循环（非递归，防栈溢出） =========================
def train(net, data, epochs, learning_rate, batch_size,
          reg_lambda=0.0, dropout_rate=0.0, lr_decay=1.0):
    """
    批量训练神经网络
    data: list of (input_vector, target_vector)
    epochs: 迭代轮数
    learning_rate: 初始学习率
    batch_size: 批大小
    reg_lambda: L2 正则系数
    dropout_rate: 隐藏层 Dropout 率
    lr_decay: 每个 epoch 后学习率衰减因子
    """
    # TODO: 添加早停 (early stopping) 机制
    for epoch in range(epochs):
        random.shuffle(data)
        total_loss = 0.0
        n_batches = 0
        for start in range(0, len(data), batch_size):
            batch = data[start:start + batch_size]
            X_batch = [x for x, _ in batch]
            Y_batch = [y for _, y in batch]

            cache, pred = forward(net, X_batch, dropout_rate)
            # MSE 损失
            loss = sum(
                sum((pred[i][j] - Y_batch[i][j]) ** 2 for j in range(len(pred[0])))
                for i in range(len(pred))
            ) / len(pred)
            total_loss += loss
            n_batches += 1

            backward(net, X_batch, Y_batch, cache, learning_rate, reg_lambda)

        avg_loss = total_loss / n_batches
        learning_rate *= lr_decay  # 学习率衰减
        print(f"Epoch {epoch + 1:3d}/{epochs}, loss: {avg_loss:.6f}")


# ========================= 普通预测 =========================
def predict(net, X):
    """前向传播，无 Dropout"""
    _, output = forward(net, X, dropout_rate=0.0)
    return output


# ========================= MC Dropout 深度思考预测 =========================
def _mc_dropout_collect(net, X, depth, dropout_rate):
    """
    递归收集带 Dropout 的输出列表
    深度由 depth 控制，防止无限递归
    """
    if depth <= 0:
        return []
    _, out = forward(net, X, dropout_rate)
    rest = _mc_dropout_collect(net, X, depth - 1, dropout_rate)
    return [out] + rest


def deep_think_predict(net, X, num_samples=10, dropout_rate=0.2):
    """
    MC Dropout 预测：运行多次带 Dropout 的前向传播并平均结果
    可用于模拟“深度思考”中的不确定性
    """
    # 至少采样一次
    if num_samples < 1:
        return predict(net, X)
    samples = _mc_dropout_collect(net, X, num_samples, dropout_rate)
    # 计算平均值
    batch_size = len(X)
    out_dim = len(samples[0][0])
    average = [[0.0] * out_dim for _ in range(batch_size)]
    for out in samples:
        for i in range(batch_size):
            for j in range(out_dim):
                average[i][j] += out[i][j]
    for i in range(batch_size):
        for j in range(out_dim):
            average[i][j] /= num_samples
    return average


# ========================= 在线学习（持续更新） =========================
def online_learn(net, x, y, learning_rate=0.01):
    """单样本在线学习，更新一次权重"""
    # 构造 batch_size=1 的数据
    X = [x]
    Y = [y]
    cache, _ = forward(net, X, dropout_rate=0.0)
    backward(net, X, Y, cache, learning_rate, reg_lambda=0.0)


# ========================= 测试与验证 =========================
def run_tests():
    """可测试空间：验证核心函数的正确性"""
    print("开始运行测试...")

    # 1. 激活函数测试
    assert abs(relu(5.0) - 5.0) < 1e-8
    assert abs(relu(-3.0) - 0.0) < 1e-8
    assert relu_derivative(1.0) == 1.0
    assert relu_derivative(0.0) == 0.0

    assert abs(sigmoid(0.0) - 0.5) < 1e-8
    assert 0.0 < sigmoid(100.0) < 1.0
    assert abs(sigmoid_derivative(0.5) - 0.25) < 1e-8

    # 2. 矩阵乘法测试
    A = [[1.0, 2.0], [3.0, 4.0]]
    B = [[1.0, 0.0], [0.0, 1.0]]
    C = matmul(A, B)
    assert C == [[1.0, 2.0], [3.0, 4.0]]

    # 3. 网络初始化形状
    net = initialize_network([2, 4, 1], [0, 1])
    assert len(net['w0']) == 2 and len(net['w0'][0]) == 4
    assert len(net['b0']) == 4
    assert len(net['w1']) == 4 and len(net['w1'][0]) == 1
    assert len(net['b1']) == 1

    # 4. 前向传播输出形状
    X = [[0.0, 0.0], [1.0, 1.0]]
    _, out = forward(net, X, dropout_rate=0.5)
    assert len(out) == 2 and len(out[0]) == 1

    # 5. 训练一轮不崩溃
    data = [([0.0, 0.0], [0.0]), ([0.0, 1.0], [1.0]), ([1.0, 0.0], [1.0]), ([1.0, 1.0], [0.0])]
    train(net, data, epochs=1, learning_rate=0.3, batch_size=4)

    # 6. 普通预测与深度思考预测不崩溃
    pred = predict(net, [[0.0, 0.0]])
    deep = deep_think_predict(net, [[0.0, 0.0]], num_samples=5, dropout_rate=0.3)
    assert len(pred[0]) == 1
    assert len(deep[0]) == 1

    # 7. 在线学习不崩溃
    online_learn(net, [0.5, 0.5], [0.5])

    print("所有测试通过！")


# ========================= 主程序：演示 XOR 问题 =========================
def main():
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        run_tests()
        return

    # 设置随机种子便于复现
    random.seed(42)

    # 构建 3 层网络：2 -> 8 -> 8 -> 1，隐藏层用 ReLU，输出用 Sigmoid
    net = initialize_network([2, 8, 8, 1], [0, 0, 1])

    xor_data = [
        ([0.0, 0.0], [0.0]),
        ([0.0, 1.0], [1.0]),
        ([1.0, 0.0], [1.0]),
        ([1.0, 1.0], [0.0])
    ]

    print("=== 训练 XOR 网络（批处理 + Dropout + L2 正则） ===")
    train(net, xor_data, epochs=2000, learning_rate=0.3, batch_size=4,
          reg_lambda=1e-5, dropout_rate=0.15, lr_decay=0.999)

    print("\n=== 普通预测 ===")
    for x, _ in xor_data:
        pred = predict(net, [x])[0][0]
        print(f"输入 {x} -> {pred:.4f}")

    print("\n=== MC Dropout 深度思考预测 (采样 30 次) ===")
    for x, _ in xor_data:
        deep = deep_think_predict(net, [x], num_samples=30, dropout_rate=0.2)[0][0]
        print(f"输入 {x} -> {deep:.4f}")

    print("\n=== 在线学习新样本 ===")
    new_samples = [([0.9, 0.1], [0.8]), ([0.2, 0.8], [0.7])]
    for x, y in new_samples:
        before = predict(net, [x])[0][0]
        online_learn(net, x, y, learning_rate=0.1)
        after = predict(net, [x])[0][0]
        print(f"样本 {x} 目标 {y[0]}：前 {before:.4f} -> 后 {after:.4f}")


if __name__ == "__main__":
    main()

# ============================================================
# Author: ciain
# Date: 2026-05-29 15:43:21
# ============================================================