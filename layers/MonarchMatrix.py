import math
from einops import rearrange
import torch
from torch import nn
import torch.nn.functional as F
from math import sqrt

from models.TimesNet import FFT_for_Period, FFT_for_Period_factor


def blockdiag_matmul(x, w):
    return torch.einsum(
        "bnm,...bm->...bn", w, x.view(*x.shape[: -1], w.shape[0], w.shape[-1])
    ).reshape(*x.shape)


def padding(x, d, dim=-1):
    if dim == -1:
        padding = torch.zeros(x.shape[0], x.shape[1], d - x.shape[-1]).to(x.device)
    elif dim == 1:
        padding = torch.zeros(x.shape[0], d - x.shape[1], x.shape[-1]).to(x.device)
    else:
        raise NotImplementedError
    return torch.cat([x, padding], dim=dim)


class Conv1D(nn.Module):
    def __init__(self, out_dim, rf, in_dim):
        super(Conv1D, self).__init__()
        self.rf = rf
        self.out_dim = out_dim
        if rf == 1:
            w = torch.empty(in_dim, out_dim)
            nn.init.normal_(w, std=0.02)
            self.w = nn.Parameter(w)
            self.b = nn.Parameter(torch.zeros(out_dim))
        else:
            raise NotImplementedError

    def forward(self, x):
        if self.rf == 1:
            # size_out = x.size()[:-1] + (self.out_dim,)
            # x = torch.addmm(self.b, x.view(-1, x.size(-1)), self.w)
            if x.dim() == 2 and self.b is not None:
                # fused op is marginally faster
                x = torch.addmm(self.b, x, self.w.t())
            else:
                output = x.matmul(self.w.t())
                if self.b is not None:
                    output += self.b
                x = output
            # x = x.view(*size_out)
        else:
            raise NotImplementedError
        return x


class MonarchMatrix(nn.Module):
    def __init__(self, sqrt_n: int):
        super().__init__()
        self.sqrt_n = sqrt_n
        self.L = nn.Parameter(torch.Tensor(sqrt_n, sqrt_n, sqrt_n))
        self.R = nn.Parameter(torch.Tensor(sqrt_n, sqrt_n, sqrt_n))

    def forward(self, x):
        x = rearrange(x, "... (m n) -> ... (n m)", n=self.sqrt_n)
        x = blockdiag_matmul(x, self.L)
        x = rearrange(x, "... (m n) -> ... (n m)", n=self.sqrt_n)
        x = blockdiag_matmul(x, self.R)
        return rearrange(x, " ... (m n) -> ... (n m)", n=self.sqrt_n)


class MonarchOutProjection(nn.Module):
    def __init__(self, d_in: int, d_out: int, n_heads: int):
        super().__init__()
        self.sqrt_d = math.ceil(math.sqrt(d_out))
        self.proj_type = "Monarch"
        if d_in * n_heads == d_out:
            self.projection = MonarchMatrix(self.sqrt_d)
            self.proj_type = "Monarch"
        elif d_in == d_out:
            self.projection = nn.ModuleList([MonarchMatrix(self.sqrt_d) for _ in range(n_heads)])
            self.proj_type = "MonarchList"
        else:
            self.projection = nn.Linear(d_in, d_out * n_heads)
            self.proj_type = "Linear"
            print("Input and output dimensions are not equal in projection, Monarch cannot be used.")
        self.n_heads = n_heads
        self.d_out = d_out

    def forward(self, x):  # x.shape = (b, n, h, d)
        b, n, hd = x.shape
        if self.proj_type == "Linear":
            x = self.projection(x)
        elif self.proj_type == "Monarch":
            if hd != self.sqrt_d ** 2:
                # 如果输入的维度不是sqrt_d的平方，则补0
                x = padding(x, self.sqrt_d ** 2, -1)
            x = self.projection(x)
        elif self.proj_type == "MonarchList":
            x = x.view(b, n, self.n_heads, -1)
            d = x.shape[-1]
            if d != self.sqrt_d ** 2:
                # 如果输入的维度不是sqrt_d的平方，则补0
                x = padding(x, self.sqrt_d ** 2, -1)
            for i in range(self.n_heads):
                x[:, :, i, :] = self.projection[i](x[:, :, i, :])
            x = torch.sum(x, dim=-2)  # [b, n, d]
        return x[..., :self.d_out]


class MonarchProjection(nn.Module):
    def __init__(self, n: int, d_in: int, d_out: int, n_heads: int):
        super().__init__()
        self.sqrt_d = math.ceil(math.sqrt(d_in))
        # self.sqrt_n = math.ceil(math.sqrt(n))
        self.proj_type = "Monarch"
        if d_in == d_out * n_heads:
            self.m1 = MonarchMatrix(self.sqrt_d)
            # self.m2 = MonarchMatrix(self.sqrt_d)
            self.b = nn.Parameter(torch.Tensor(self.sqrt_d ** 2))
            self.d_kernel = nn.Parameter(torch.Tensor(n, self.sqrt_d ** 2))
            self.proj_type = "Monarch"
        elif d_in == d_out:
            self.projection = nn.ModuleList([MonarchMatrix(self.sqrt_d) for _ in range(n_heads)])
            self.proj_type = "MonarchList"
        else:
            self.projection = nn.Linear(d_in, d_out * n_heads)
            self.proj_type = "Linear"
            print("Input and output dimensions are not equal in projection, Monarch cannot be used.")
        self.n_heads = n_heads
        self.d_out = d_out
        self.d_in = d_in
        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.proj_type == "Monarch":
            nn.init.kaiming_uniform_(self.d_kernel, a=math.sqrt(5))
            bound = 1 / math.sqrt(self.d_in)
            nn.init.uniform_(self.b, -bound, bound)
            bound = 1
            # nn.init.kaiming_uniform_(self.m1.L, a=math.sqrt(5))
            # nn.init.kaiming_uniform_(self.m1.R, a=math.sqrt(5))
            nn.init.uniform_(self.m1.L, -bound, bound)
            nn.init.uniform_(self.m1.R, -bound, bound)
            # nn.init.uniform_(self.m2.L, -bound, bound)
            # nn.init.uniform_(self.m2.R, -bound, bound)

    def forward(self, x):
        b, n, d = x.shape
        # print("n=",n)
        if self.proj_type == "Linear":
            x = self.projection(x)
        else:
            if d != self.sqrt_d ** 2:
                # 如果输入的维度不是sqrt_d的平方，则补0
                x = padding(x, self.sqrt_d ** 2, -1)
            if self.proj_type == "Monarch":
                # x = self.m1(x) + self.b
                # print(d, self.sqrt_d)
                x = self.d_kernel[:n, ...] * self.m1(x) + self.b  # best for now
                # x = self.m2(self.d_kernel * self.m1(x)) + self.b
            elif self.proj_type == "MonarchList":
                x = torch.stack([self.projection[i](x) for i in range(self.n_heads)], dim=-1)
                x = x.view(b, n, -1)
        return x[..., :self.d_out * self.n_heads]  # [b, n, h * d]


class MonarchAttention(nn.Module):
    def __init__(self, output_attention=False, sqrt_n=None, d_model=None, dropout=None):
        super().__init__()
        self.output_attention = output_attention

        self.sqrt_n = sqrt_n
        self.sqrt_d = math.ceil(math.sqrt(d_model))
        # self.n_components = n_components
        # self.M1 = nn.ModuleList([MonarchMatrix(self.sqrt_n) for _ in range(n_components)])
        # self.M2 = nn.ModuleList([MonarchMatrix(self.sqrt_n) for _ in range(n_components)])
        self.M1 = MonarchMatrix(self.sqrt_n)
        self.M2 = MonarchMatrix(self.sqrt_n)

        # self.K = nn.Parameter(torch.Tensor(n_components, d_model, self.sqrt_n ** 2))
        # self.b1 = nn.Parameter(torch.Tensor(n_components, self.sqrt_n ** 2))
        # self.b2 = nn.Parameter(torch.Tensor(n_components, self.sqrt_n ** 2))
        self.b1 = nn.Parameter(torch.Tensor(self.sqrt_n ** 2))
        self.b2 = nn.Parameter(torch.Tensor(self.sqrt_n ** 2))

        self.dropout = dropout

        # following paras need to be set outside, i.e. AttentionLayer
        self.reset_parameters()

    def reset_parameters(self) -> None:
        bound = 1 / self.sqrt_n
        nn.init.uniform_(self.b1, -bound, bound)
        nn.init.uniform_(self.b2, -bound, bound)

        bound = 1
        a = math.sqrt(5)
        # nn.init.kaiming_uniform_(self.K, a=a)
        # nn.init.uniform_(self.M1.L, -bound, bound)
        # nn.init.uniform_(self.M1.R, -bound, bound)
        nn.init.kaiming_uniform_(self.M1.L, a=a)
        nn.init.kaiming_uniform_(self.M1.R, a=a)
        # nn.init.uniform_(self.M2.L, -bound, bound)
        # nn.init.uniform_(self.M2.R, -bound, bound)
        nn.init.kaiming_uniform_(self.M2.L, a=a)
        nn.init.kaiming_uniform_(self.M2.R, a=a)

    def forward(self, queries, keys, values):
        """
        Monarch Attention, only used in self-attention, not cross-attention.
        V * M2(M1(Q * K))
        V * M2(K * M1Q)
        """
        B, S, H, D = queries.shape
        queries = queries.permute(0, 2, 3, 1)  # [B, H, D, S]
        keys = keys.permute(0, 2, 3, 1)  # [B, H, D, S]
        values = values.permute(0, 2, 3, 1)  # [B, H, D, S]

        queries = F.pad(queries, (0, self.sqrt_n ** 2 - S), value=0)
        keys = F.pad(keys, (0, self.sqrt_n ** 2 - S), value=0)
        # values = F.pad(values, (0, self.sqrt_n ** 2 - S), value=0)

        # A = self.K[i] * self.M1[i](queries * keys)
        # queries_i = rearrange(queries, "... (p f) -> ... (f p)", p=periods[i])
        # keys_i = rearrange(keys, "... (p f) -> ... (f p)", p=periods[i])
        # values_i = rearrange(values, "... (p f) -> ... (f p)", p=periods[i])

        A = self.dropout(self.M1(queries) * keys) + self.b1
        # A = F.gelu(A)
        A = self.M2(A) + self.b2
        # A = self.dropout(torch.softmax(A, dim=-1))
        # A = self.dropout(F.gelu(A))
        # A = self.dropout(self.M1[i](queries * keys + self.b1[i]))
        # A = self.M2[i](A) + self.b2[i]

        Y = A[..., :S] * values  # [B, H, D, S]
        # y = rearrange(y, "... (p f) -> ... (f p)", p=periods[i])
        # Y.append(y)

        # if self.n_components == 1:
        #     # without adaptive aggregation
        #     Y = Y[0]  # [B, H, D, S]
        # else:
        #     # adaptive aggregation
        #     Y = torch.stack(Y, dim=-1)  # [B, H, D, S, k]
        #     # weights = F.softmax(weights, dim=1)
        #     # weights = weights.unsqueeze(1).unsqueeze(1).unsqueeze(1).repeat(1, H, D, S, 1)  # [B, H, D, S, k]
        #     Y = torch.sum(Y * weights, -1)  # [B, H, D, S]

        # Y = self.dropout(Y)
        return Y, None


class MonarchScoreCalculator(nn.Module):
    def __init__(self, n_components: int, sqrt_n: int):
        """This can only be used in self-attention, cannot be used in cross-attention."""
        super().__init__()
        self.sqrt_n = sqrt_n
        self.n_components = n_components
        self.M1 = nn.ModuleList([MonarchMatrix(self.sqrt_n) for _ in range(n_components)])
        self.M2 = nn.ModuleList([MonarchMatrix(self.sqrt_n) for _ in range(n_components)])

        self.reset_parameters()

    def reset_parameters(self) -> None:
        bound = 1
        for m in self.M1:
            nn.init.uniform_(m.L, -bound, bound)
            nn.init.uniform_(m.R, -bound, bound)
            # nn.init.kaiming_uniform_(m.L, a=math.sqrt(5))
            # nn.init.kaiming_uniform_(m.R, a=math.sqrt(5))
        for m in self.M2:
            nn.init.uniform_(m.L, -bound, bound)
            nn.init.uniform_(m.R, -bound, bound)
            # nn.init.kaiming_uniform_(m.L, a=math.sqrt(5))
            # nn.init.kaiming_uniform_(m.R, a=math.sqrt(5))

    def forward(self, queries, keys, periods, weights):
        """
        :param queries: [B, S, H, D]
        :param keys: [B, L, H, D]
        :param periods: [k]
        :param weights: [B, k]
        :return: [B, H, S, L]
        """
        B, S, H, D = queries.shape
        queries = queries.permute(0, 2, 3, 1)  # [B, H, D, S]
        keys = keys.permute(0, 2, 3, 1)  # [B, H, D, L]
        queries = F.pad(queries, (0, self.sqrt_n ** 2 - S), mode='constant', value=0)
        keys = F.pad(keys, (0, self.sqrt_n ** 2 - S), mode='constant', value=0)
        score_list = []
        for i in range(self.n_components):
            period = periods[i]

            queries_i = rearrange(queries, "... (p f) -> ... (f p)", p=period)
            # queries_i = queries
            queries_i = self.M1[i](queries_i)
            queries_i = rearrange(queries_i, "... (p f) -> ... (f p)", p=period)  # [B, H, D, S]

            keys_i = rearrange(keys, "... (p f) -> ... (f p)", p=period)
            # keys_i = keys
            keys_i = self.M2[i](keys_i)
            keys_i = rearrange(keys_i, "... (p f) -> ... (f p)", p=period)

            # queries_i = rearrange(queries_i[..., :S], 'b h d s -> b h d 1 s')
            # keys_i = rearrange(keys_i[..., :L], 'b h d l -> b h d l 1')
            # score_i = queries_i * keys_i  # [B, H, D, S, L]
            # score_list.append(score_i.sum(dim=-3))  # [B, H, S, L]
            score_list.append(torch.einsum("bhds,bhdl->bhsl", queries_i[..., :S], keys_i[..., :S]))  # [B, H, S, L]
        # adaptive aggregation
        scores = torch.stack(score_list, dim=-1)  # [B, H, S, S, k]
        weights = F.softmax(weights, dim=1)
        weights = weights.unsqueeze(1).unsqueeze(1).unsqueeze(1).repeat(1, H, S, S, 1)  # [B, H, S, S, k]
        scores = torch.sum(scores * weights, -1)  # [B, H, S, S]
        return scores


class MonarchFFN(nn.Module):
    def __init__(self, d_in: int, dropout=0.1, activation="relu"):
        super().__init__()
        sqrt_d = math.ceil(math.sqrt(d_in))
        self.m1 = MonarchMatrix(sqrt_d)
        self.m2 = MonarchMatrix(sqrt_d)

        self.d_kernel = nn.Parameter(torch.Tensor(1, sqrt_d ** 2))
        self.b1 = nn.Parameter(torch.Tensor(sqrt_d ** 2))
        self.b2 = nn.Parameter(torch.Tensor(sqrt_d ** 2))
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else nn.GELU()
        self.sqrt_d = sqrt_d
        self.d_in = d_in
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.d_kernel, a=math.sqrt(5))
        bound = 1 / math.sqrt(self.d_in)
        nn.init.uniform_(self.b1, -bound, bound)
        nn.init.uniform_(self.b2, -bound, bound)

        a = math.sqrt(5)
        nn.init.kaiming_uniform_(self.m1.L, a=a)
        nn.init.kaiming_uniform_(self.m1.R, a=a)
        bound = 1
        nn.init.uniform_(self.m2.L, -bound, bound)
        nn.init.uniform_(self.m2.R, -bound, bound)

    def forward(self, x: torch.Tensor):
        b, n, d = x.shape
        if d != self.sqrt_d ** 2:
            # 如果输入的维度不是sqrt_d的平方，则补0
            x = padding(x, self.sqrt_d ** 2, -1)
        y = self.activation(self.d_kernel * self.m1(x)) + self.b1
        y = self.dropout(self.m2(y)) + self.b2
        return y[..., :d]


class MonarchMixerLayer(nn.Module):
    def __init__(self, sqrt_n: int, sqrt_d: int):
        super().__init__()
        self.m1 = MonarchMatrix(sqrt_n)
        self.m2 = MonarchMatrix(sqrt_n)
        self.m3 = MonarchMatrix(sqrt_d)
        self.m4 = MonarchMatrix(sqrt_d)

        self.n_kernel = nn.Parameter(torch.randn(sqrt_d ** 2, sqrt_n ** 2))
        self.d_kernel = nn.Parameter(torch.randn(1, sqrt_d ** 2))
        self.layer_norm = nn.LayerNorm(sqrt_d ** 2)

        self.sqrt_n = sqrt_n
        self.sqrt_d = sqrt_d

    def forward(self, x: torch.Tensor):  # x. shape = (b, n, d)
        b, n, d = x.shape
        x = padding(x, self.sqrt_n ** 2, dim=1)  # padding
        x_tilde = self.m2(torch.relu(self.n_kernel * self.m1(x.transpose(-1, -2)))).transpose(-1, -2)  # mix sequence
        x_tilde = padding(x_tilde[:, :n, :], self.sqrt_d ** 2, dim=-1)
        y = self.m4(torch.relu(self.d_kernel * self.m3(x_tilde)))  # mix features
        y = y[..., :d]
        return self.layer_norm(y + x_tilde)  # skip connection


if __name__ == "__main__":
    x = torch.randn(32, 100, 512)
    B, N, D = x.shape
    period, weight = FFT_for_Period_factor(x, N, 3)
    print(period.shape)
    print(weight.shape)
    print(weight)
    y = torch.randn(32, 8, 64, 96, 3)
    y = torch.sum(y * weight, -1)
    print(y.shape)
    """
    q = torch.randn(32, 8, 64, 96)  # [b, h, d, s]
    k = torch.randn(32, 8, 64, 96)
    q = rearrange(q, 'b h d s -> b h d 1 s')
    k = rearrange(k, 'b h d l -> b h d l 1')
    qk = q * k
    print(qk.shape)
    qk = qk.sum(dim=-3)
    print(qk.shape)
    """
    """
    x = torch.randn(64, 96, 512)
    B, N, D = x.shape
    L = N
    sqrt_D = math.ceil(math.sqrt(D))
    k = 3
    n_heads = 8
    # 向上取整
    sqrt_n = math.ceil(math.sqrt(N))
    new_N = sqrt_n ** 2
    new_D = sqrt_D ** 2

    # layers
    query_projection = MonarchProjection(D, D, n_heads)
    x = query_projection(x)
    key_projection = MonarchProjection(sqrt_D, n_heads)
    value_projection = MonarchProjection(sqrt_D, n_heads)
    out_projection = MonarchOutProjection(sqrt_D, n_heads)
    score_calculator = MonarchScoreCalculator(k)

    # 补0
    x = padding(x, new_N, dim=1)
    # 生成周期列表
    period_list, period_weight = FFT_for_Period_factor(x, k)

    # projection
    query = query_projection(x)  # [b, h, n, d]
    key = key_projection(x)  # [b, h, n, d]
    value = value_projection(x)

    A = score_calculator(query, key, period_list)  # [b, h, n, n]
    y = []
    for i in range(k):
        yi = torch.einsum("bhsl,blhd->bshd", A[i], value)[:, :L, :, :].contiguous()
        y.append(out_projection(yi.view(B, L, -1))[..., :D])
        # adaptive aggregation
    y = torch.stack(y, dim=-1)
    period_weight = F.softmax(period_weight, dim=1)
    period_weight = period_weight.unsqueeze(1).unsqueeze(1).repeat(1, L, D, 1)
    y = torch.sum(y * period_weight, -1)
    print(sqrt_n)
    print(x.shape)
    """
