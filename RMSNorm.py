import torch
import torch.nn as nn
import torch.nn.functional as F
import math

"""
nn.Parameter：这是 PyTorch 中的一个类，用于定义模型中的可训练参数。
当将一个张量包装在 nn.Parameter 中时，PyTorch 会将其自动添加到模型的参数列表中，便于在优化过程中更新。
"""

class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super(RMSNorm, self).__init__()
        """
        LLaMaRMSNorm is equivalent to T5LayerNorm
        """
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps  # eps 防止取倒数之后分母为 0

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        # torch.rsqrt 计算每个元素的平方根的倒数
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        # weight 是末尾乘的可训练参数, 即 g_i
        return (self.weight * hidden_states).to(input_dtype)



