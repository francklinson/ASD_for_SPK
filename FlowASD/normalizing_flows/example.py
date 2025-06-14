import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt


# 1. 定义耦合层（Coupling Layer）
class CouplingLayer(nn.Module):
    def __init__(self, dim, hidden_dim=64):
        super().__init__()
        # 神经网络用于学习变换参数
        self.net = nn.Sequential(
            nn.Linear(dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, dim // 2 * 2)  # 输出平移和缩放参数
        )
        # 初始化权重
        self.net[-1].weight.data.fill_(0)
        self.net[-1].bias.data.fill_(0)

    def forward(self, x, reverse=False):
        x1, x2 = x.chunk(2, dim=1)  # 将输入切分为两部分
        st = self.net(x1)
        s, t = st.chunk(2, dim=1)  # s: 缩放, t: 平移
        s = torch.tanh(s)  # 限制缩放范围

        if not reverse:
            y2 = x2 * torch.exp(s) + t  # 正向变换
            log_det = s.sum(dim=1)  # 对数雅可比行列式
        else:
            y2 = (x2 - t) * torch.exp(-s)  # 反向变换（采样时用）
            log_det = -s.sum(dim=1)
        return torch.cat([x1, y2], dim=1), log_det


# 2. 构建 NICE 模型
class NICE(nn.Module):
    def __init__(self, dim, num_layers=4):
        super().__init__()
        self.layers = nn.ModuleList([
            CouplingLayer(dim) for _ in range(num_layers)
        ])
        # 交替交换输入维度以增强表达能力
        self.permute = lambda x: x[:, torch.randperm(x.shape[1])]

    def forward(self, x, reverse=False):
        log_det_total = 0
        if not reverse:
            for layer in self.layers:
                x, log_det = layer(x)
                log_det_total += log_det
        else:
            for layer in reversed(self.layers):
                x, log_det = layer(x, reverse=True)
                log_det_total += log_det
        return x, log_det_total

    def log_prob(self, x):
        z, log_det = self.forward(x)
        # 假设先验分布为标准正态分布
        prior_log_prob = -0.5 * (z ** 2).sum(dim=1) - 0.5 * np.log(2 * np.pi) * z.shape[1]
        return prior_log_prob + log_det


# 3. 训练流程
def train():
    dim = 2  # 数据维度
    model = NICE(dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # 生成螺旋数据集（2D）
    num_samples = 10000
    theta = np.linspace(0, 4 * np.pi, num_samples)
    x = np.stack([theta * np.cos(theta), theta * np.sin(theta)], axis=1)
    x = (x - x.mean(0)) / x.std(0)  # 标准化
    data = torch.tensor(x, dtype=torch.float32)

    # 训练循环
    for epoch in range(10000):
        idx = torch.randperm(len(data))[:256]  # Mini-batch
        batch = data[idx]
        loss = -model.log_prob(batch).mean()  # 最大化对数似然

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

    # 保存模型
    torch.save(model.state_dict(), "nice_model.pth")
    return model


# 4. 采样生成新数据
def sample(model, num_samples=1000):
    z = torch.randn(num_samples, 2)  # 从标准正态分布采样
    x, _ = model(z, reverse=True)  # 通过反向变换生成数据
    return x.detach().numpy()


# 运行示例
if __name__ == "__main__":
    model = train()
    samples = sample(model)

    # 可视化结果
    plt.scatter(samples[:, 0], samples[:, 1], s=1, alpha=0.5)
    plt.title("Generated Samples from NICE Model")
    plt.show()