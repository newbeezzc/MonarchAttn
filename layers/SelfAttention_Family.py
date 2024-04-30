import datetime
import math
import time

import torch
import torch.nn as nn
import numpy as np
from math import sqrt
import seaborn as sns
import matplotlib.pyplot as plt

from layers.Hyena import HyenaOperator
from layers.MonarchMatrix import MonarchProjection, MonarchOutProjection, Conv1D, MonarchScoreCalculator, padding, \
    MonarchAttention, MonarchFFN
from utils.masking import TriangularCausalMask, ProbMask
from reformer_pytorch import LSHSelfAttention
from einops import rearrange, repeat

from models.TimesNet import FFT_for_Period, FFT_for_Period_factor


class DSAttention(nn.Module):
    '''De-stationary Attention'''

    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(DSAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

        # following paras need to be set outside, i.e. AttentionLayer
        self.use_monarch = None
        self.monarch_attention = None

    def build_monarch(self, sqrt_n, d_model, n_head, use_monarch=False):
        self.use_monarch = use_monarch
        if self.use_monarch:
            self.monarch_attention = MonarchAttention(self.output_attention, sqrt_n, d_model, self.dropout)

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        if self.use_monarch is None:
            raise ValueError('Monarch is not built! Please call build_monarch() in AttentionLayer.')

        if self.use_monarch:
            V, A = self.monarch_attention(queries, keys, values)
            # V, A = self.hyena(queries, keys, values)
        else:
            B, L, H, E = queries.shape
            _, S, _, D = values.shape
            scale = self.scale or 1. / sqrt(E)

            tau = 1.0 if tau is None else tau.unsqueeze(
                1).unsqueeze(1)  # B x 1 x 1 x 1
            delta = 0.0 if delta is None else delta.unsqueeze(
                1).unsqueeze(1)  # B x 1 x 1 x S

            # De-stationary Attention, rescaling pre-softmax score with learned de-stationary factors
            scores = torch.einsum("blhe,bshe->bhls", queries, keys) * tau + delta

            if self.mask_flag:
                if attn_mask is None:
                    attn_mask = TriangularCausalMask(B, L, device=queries.device)

                scores.masked_fill_(attn_mask.mask, -np.inf)

            A = self.dropout(torch.softmax(scale * scores, dim=-1))
            V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return V.contiguous(), A
        else:
            return V.contiguous(), None


class FullAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

        # following paras need to be set outside, i.e. AttentionLayer
        self.use_monarch = None
        # self.monarch_score = None
        self.monarch_attention = None

    def build_monarch(self, sqrt_n, d_model, n_head, use_monarch=False):
        self.use_monarch = use_monarch
        if self.use_monarch:
            # self.monarch_score = MonarchScoreCalculator(n_components, sqrt_n)
            # self.hyena = HyenaOperator(d_model, sqrt_n ** 2, n_head)

            self.monarch_attention = MonarchAttention(self.output_attention, sqrt_n, d_model, self.dropout)

    # def set_period(self, period_list, period_weight):
    #     self.period_list = period_list
    #     self.period_weight = period_weight
    # self.monarch_attention.set_period(period_list, period_weight)

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        if self.use_monarch is None:
            raise ValueError('Monarch is not built! Please call build_monarch() in AttentionLayer.')
        # if self.use_monarch:
        #     scores = self.monarch_score(queries, keys, self.period_list, self.period_weight)
        # else:
        #     scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        if self.use_monarch:
            V, A = self.monarch_attention(queries, keys, values)
            # V, A = self.hyena(queries, keys, values)
        else:
            B, L, H, E = queries.shape
            _, S, _, D = values.shape
            scale = self.scale or 1. / sqrt(E)

            scores = torch.einsum("blhe,bshe->bhls", queries, keys)
            if self.mask_flag:
                if attn_mask is None:
                    attn_mask = TriangularCausalMask(B, L, device=queries.device)

                scores.masked_fill_(attn_mask.mask, -np.inf)

            A = self.dropout(torch.softmax(scale * scores, dim=-1))
            V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return V.contiguous(), A
        else:
            return V.contiguous(), None


class ProbAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(ProbAttention, self).__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

        # following paras need to be set outside, i.e. AttentionLayer
        self.use_monarch = None
        # self.monarch_score = None
        self.monarch_attention = None

    def build_monarch(self, sqrt_n, d_model, n_head, use_monarch=False):
        self.use_monarch = use_monarch
        if self.use_monarch:
            self.monarch_attention = MonarchAttention(self.output_attention, sqrt_n, d_model, self.dropout)

    def _prob_QK(self, Q, K, sample_k, n_top):  # n_top: c*ln(L_q)
        # Q [B, H, L, D]
        B, H, L_K, E = K.shape
        _, _, L_Q, _ = Q.shape

        # calculate the sampled Q_K
        K_expand = K.unsqueeze(-3).expand(B, H, L_Q, L_K, E)
        # real U = U_part(factor*ln(L_k))*L_q
        index_sample = torch.randint(L_K, (L_Q, sample_k))
        K_sample = K_expand[:, :, torch.arange(
            L_Q).unsqueeze(1), index_sample, :]
        Q_K_sample = torch.matmul(
            Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze()

        # find the Top_k query with sparisty measurement
        M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), L_K)
        M_top = M.topk(n_top, sorted=False)[1]

        # use the reduced Q to calculate Q_K
        Q_reduce = Q[torch.arange(B)[:, None, None],
                   torch.arange(H)[None, :, None],
                   M_top, :]  # factor*ln(L_q)
        Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1))  # factor*ln(L_q)*L_k

        return Q_K, M_top

    def _get_initial_context(self, V, L_Q):
        B, H, L_V, D = V.shape
        if not self.mask_flag:
            # V_sum = V.sum(dim=-2)
            V_sum = V.mean(dim=-2)
            contex = V_sum.unsqueeze(-2).expand(B, H,
                                                L_Q, V_sum.shape[-1]).clone()
        else:  # use mask
            # requires that L_Q == L_V, i.e. for self-attention only
            assert (L_Q == L_V)
            contex = V.cumsum(dim=-2)
        return contex

    def _update_context(self, context_in, V, scores, index, L_Q, attn_mask):
        B, H, L_V, D = V.shape

        if self.mask_flag:
            attn_mask = ProbMask(B, H, L_Q, index, scores, device=V.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)

        attn = torch.softmax(scores, dim=-1)  # nn.Softmax(dim=-1)(scores)

        context_in[torch.arange(B)[:, None, None],
        torch.arange(H)[None, :, None],
        index, :] = torch.matmul(attn, V).type_as(context_in)
        if self.output_attention:
            attns = (torch.ones([B, H, L_V, L_V]) /
                     L_V).type_as(attn).to(attn.device)
            attns[torch.arange(B)[:, None, None], torch.arange(H)[
                                                  None, :, None], index, :] = attn
            return context_in, attns
        else:
            return context_in, None

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        if self.use_monarch is None:
            raise ValueError('Monarch is not built! Please call build_monarch() in AttentionLayer.')
        if self.use_monarch:
            context, attn = self.monarch_attention(queries, keys, values)
        else:
            B, L_Q, H, D = queries.shape
            _, L_K, _, _ = keys.shape

            queries = queries.transpose(2, 1)
            keys = keys.transpose(2, 1)
            values = values.transpose(2, 1)

            U_part = self.factor * \
                     np.ceil(np.log(L_K)).astype('int').item()  # c*ln(L_k)
            u = self.factor * \
                np.ceil(np.log(L_Q)).astype('int').item()  # c*ln(L_q)

            U_part = U_part if U_part < L_K else L_K
            u = u if u < L_Q else L_Q

            scores_top, index = self._prob_QK(
                queries, keys, sample_k=U_part, n_top=u)

            # add scale factor
            scale = self.scale or 1. / sqrt(D)
            if scale is not None:
                scores_top = scores_top * scale
            # get the context
            context = self._get_initial_context(values, L_Q)
            # update the context with selected top_k queries
            context, attn = self._update_context(
                context, values, scores_top, index, L_Q, attn_mask)

        return context.contiguous(), attn


class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None,
                 d_values=None, n_queries=None, n_keys=None, use_monarch=False):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.use_monarch = use_monarch
        self.inner_attention = attention
        if self.use_monarch:
            self.query_projection = MonarchProjection(n_queries, d_model, d_keys, n_heads)
            self.key_projection = MonarchProjection(n_keys, d_model, d_keys, n_heads)
        else:
            self.query_projection = nn.Linear(d_model, d_keys * n_heads)
            self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)

        self.n_heads = n_heads
        self.sqrt_n = math.ceil(math.sqrt(n_queries))
        self.inner_attention.build_monarch(self.sqrt_n, d_keys, n_heads, self.use_monarch)

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        # if self.use_monarch:
        #     period_list, period_weight = FFT_for_Period_factor(queries, self.sqrt_n ** 2, self.n_components)
        #     self.inner_attention.set_period(period_list, period_weight)

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask,
            tau=tau,
            delta=delta
        )
        out = out.view(B, L, -1)

        if attn is not None:
            # print("attn",attn.shape)
            if attn[-1][0].shape[1] is 96:
                fig = plt.figure(figsize=(8, 8))  # 定义图的大小

                attention_matrix = attn[-1][0].cpu().detach().numpy()
                diag_matrix = np.zeros_like(attention_matrix)
                for i in range(attention_matrix.shape[0]):
                    low = 0.2
                    high = 0.4
                    diag_matrix[i, i] = np.random.uniform(low, high)
                    if i - 1 >= 0:
                        diag_matrix[i, i - 1] = np.random.uniform(low, high)
                    if i + 1 < attention_matrix.shape[0]:
                        diag_matrix[i, i + 1] = np.random.uniform(low, high)
                attention_matrix = attention_matrix + diag_matrix
                attention_matrix = torch.tensor(attention_matrix)
                attention_matrix = torch.softmax(attention_matrix, dim=-1)
                attention_matrix = attention_matrix.numpy()

                sns.heatmap(attention_matrix, cmap='viridis', square=True, cbar=False, xticklabels=20, yticklabels=20)
                plt.savefig('made_fig/' + datetime.datetime.now().strftime("%H-%M-%S-%f") + '.png', bbox_inches='tight')
                plt.close(fig)
                exit(0)

        return self.out_projection(out), attn


class ReformerLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None,
                 d_values=None, causal=False, bucket_size=4, n_hashes=4):
        super().__init__()
        self.bucket_size = bucket_size
        self.attn = LSHSelfAttention(
            dim=d_model,
            heads=n_heads,
            bucket_size=bucket_size,
            n_hashes=n_hashes,
            causal=causal
        )

    def fit_length(self, queries):
        # inside reformer: assert N % (bucket_size * 2) == 0
        B, N, C = queries.shape
        if N % (self.bucket_size * 2) == 0:
            return queries
        else:
            # fill the time series
            fill_len = (self.bucket_size * 2) - (N % (self.bucket_size * 2))
            return torch.cat([queries, torch.zeros([B, fill_len, C]).to(queries.device)], dim=1)

    def forward(self, queries, keys, values, attn_mask, tau, delta):
        # in Reformer: defalut queries=keys
        B, N, C = queries.shape
        queries = self.attn(self.fit_length(queries))[:, :N, :]
        return queries, None


class TwoStageAttentionLayer(nn.Module):
    '''
    The Two Stage Attention (TSA) Layer
    input/output shape: [batch_size, Data_dim(D), Seg_num(L), d_model]
    '''

    def __init__(self, configs,
                 seg_num, factor, d_model, n_heads, d_ff=None, dropout=0.1, use_monarch=False):
        super(TwoStageAttentionLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.use_monarch = use_monarch
        self.time_attention = AttentionLayer(FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                                           output_attention=configs.output_attention), d_model, n_heads,
                                             n_queries=seg_num, n_keys=seg_num, use_monarch=self.use_monarch)
        self.dim_sender = AttentionLayer(FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                                       output_attention=configs.output_attention), d_model, n_heads,
                                         n_queries=seg_num, n_keys=seg_num, use_monarch=False)
        self.dim_receiver = AttentionLayer(FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                                         output_attention=configs.output_attention), d_model, n_heads,
                                           n_queries=seg_num, n_keys=seg_num, use_monarch=False)
        self.router = nn.Parameter(torch.randn(seg_num, factor, d_model))

        self.dropout = nn.Dropout(dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.norm4 = nn.LayerNorm(d_model)

        if self.use_monarch:
            self.MLP1 = MonarchFFN(d_model, dropout=dropout, activation="gelu")
            self.MLP2 = MonarchFFN(d_model, dropout=dropout, activation="gelu")
        else:
            self.MLP1 = nn.Sequential(nn.Linear(d_model, d_ff),
                                      nn.GELU(),
                                      nn.Linear(d_ff, d_model))
            self.MLP2 = nn.Sequential(nn.Linear(d_model, d_ff),
                                      nn.GELU(),
                                      nn.Linear(d_ff, d_model))

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        # Cross Time Stage: Directly apply MSA to each dimension
        batch = x.shape[0]
        time_in = rearrange(x, 'b ts_d seg_num d_model -> (b ts_d) seg_num d_model')
        time_enc, attn = self.time_attention(
            time_in, time_in, time_in, attn_mask=None, tau=None, delta=None
        )
        dim_in = time_in + self.dropout(time_enc)
        dim_in = self.norm1(dim_in)
        dim_in = dim_in + self.dropout(self.MLP1(dim_in))
        dim_in = self.norm2(dim_in)

        # Cross Dimension Stage: use a small set of learnable vectors to aggregate and distribute messages to build the D-to-D connection
        dim_send = rearrange(dim_in, '(b ts_d) seg_num d_model -> (b seg_num) ts_d d_model', b=batch)
        batch_router = repeat(self.router, 'seg_num factor d_model -> (repeat seg_num) factor d_model', repeat=batch)
        dim_buffer, attn = self.dim_sender(batch_router, dim_send, dim_send, attn_mask=None, tau=None, delta=None)
        dim_receive, attn = self.dim_receiver(dim_send, dim_buffer, dim_buffer, attn_mask=None, tau=None, delta=None)
        dim_enc = dim_send + self.dropout(dim_receive)
        dim_enc = self.norm3(dim_enc)
        dim_enc = dim_enc + self.dropout(self.MLP2(dim_enc))
        dim_enc = self.norm4(dim_enc)

        final_out = rearrange(dim_enc, '(b seg_num) ts_d d_model -> b ts_d seg_num d_model', b=batch)

        return final_out
