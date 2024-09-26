# The code is based on the PyTorch implementation:
# https://zhuanlan.zhihu.com/p/701777558
# https://github.com/mst272/LLM-Dojo/blob/main/llm_tricks/moe/make_moe_step_by_step.ipynb


import torch
from torch import nn
import torch.nn.functional as F

#Expert module
class Expert(nn.Module):
    """ An MLP is a simple linear layer followed by a non-linearity i.e. each Expert """

    def __init__(self, n_embd):
        super().__init__()
        self.dropout = 0.1
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(self.dropout),
        )

    def forward(self, x):
        return self.net(x)

class NoisyTopkRouter(nn.Module):
    def __init__(self, n_embed, num_experts, top_k):
        super(NoisyTopkRouter, self).__init__()
        self.top_k = top_k
        self.topkroute_linear = nn.Linear(n_embed, num_experts)
        # add noise
        self.noise_linear = nn.Linear(n_embed, num_experts)

    def forward(self, mh_output):
        logits = self.topkroute_linear(mh_output)

        # Noise logits
        noise_logits = self.noise_linear(mh_output)

        # Adding scaled unit gaussian noise to the logits
        noise = torch.randn_like(logits) * F.softplus(noise_logits)
        noisy_logits = logits + noise

        top_k_logits, indices = noisy_logits.topk(self.top_k, dim=-1)
        zeros = torch.full_like(noisy_logits, float('-inf'))
        sparse_logits = zeros.scatter(-1, indices, top_k_logits)
        router_output = F.softmax(sparse_logits, dim=-1)
        return router_output, indices


class NoisyTopkRouter_Cluster(nn.Module):
    def __init__(self, top_k):
        super(NoisyTopkRouter_Cluster, self).__init__()
        self.top_k = top_k

    def forward(self, logits):
        # Adding scaled unit gaussian noise to the logits
        noise = torch.randn_like(logits) * F.softplus(logits)
        noisy_logits = logits + noise

        top_k_logits, indices = noisy_logits.topk(self.top_k, dim=-1)
        zeros = torch.full_like(noisy_logits, float('-inf'))
        sparse_logits = zeros.scatter(-1, indices, top_k_logits)
        router_output = F.softmax(sparse_logits, dim=-1)
        return router_output, indices


class SparseMoE(nn.Module):
    def __init__(self, n_embed, num_experts, top_k):
        super(SparseMoE, self).__init__()
        self.router = NoisyTopkRouter_Cluster(top_k)
        self.experts = nn.ModuleList([Expert(n_embed) for _ in range(num_experts)])
        self.top_k = top_k

    def forward(self, x, expert):
        # print(x.shape)
        # 1. 输入进入router得到两个输出
        gating_output, indices = self.router(expert)
        # 2.初始化全零矩阵，后续叠加为最终结果
        final_output = torch.zeros_like(x)

        # 3.展平，即把每个batch拼接到一起，这里对输入x和router后的结果都进行了展平
        flat_x = x.reshape(-1, x.size(-1))
        # print(flat_x.shape, x.shape)    # torch.Size([5376, 112]) torch.Size([128, 42, 112])
        flat_gating_output = gating_output.view(-1, gating_output.size(-1))
        # print(flat_gating_output.shape) # torch.Size([5376, 2])

        # 以每个专家为单位进行操作，即把当前专家处理的所有token都进行加权
        for i, expert in enumerate(self.experts):
            # 4. 对当前的专家(例如专家0)来说，查看其对所有tokens中哪些在前top2
            expert_mask = (indices == i).any(dim=-1)
            # 5. 展平操作
            flat_mask = expert_mask.view(-1)
            # 如果当前专家是任意一个token的前top2
            if flat_mask.any():
                # 6. 得到该专家对哪几个token起作用后，选取token的维度表示
                expert_input = flat_x[flat_mask]
                # 7. 将token输入expert得到输出
                expert_output = expert(expert_input)

                # 8. 计算当前专家对于有作用的token的权重分数
                gating_scores = flat_gating_output[flat_mask, i].unsqueeze(1)
                # 9. 将expert输出乘上权重分数
                weighted_output = expert_output * gating_scores

                # 10. 循环进行做种的结果叠加
                # print(weighted_output.shape)    # torch.Size([2643, 112])
                flat_final_output = final_output.reshape(-1, final_output.size(-1))
                flat_final_output[expert_mask] += weighted_output.squeeze(1)
                final_output = flat_final_output.reshape(x.shape[0], x.shape[1], x.shape[2])
                # print(final_output.shape)

        return final_output
