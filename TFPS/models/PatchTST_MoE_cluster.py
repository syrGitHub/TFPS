__all__ = ['PatchTST']

# Cell
from typing import Callable, Optional
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
import numpy as np

from layers.PatchTST_MoE_backbone_cluster_time import PatchTST_MoE_cluster_time
from layers.PatchTST_MoE_backbone_cluster_frequency import PatchTST_MoE_cluster_frequency
from layers.PatchTST_layers import series_decomp
from layers.RevIN import RevIN

class Flatten_Head(nn.Module):
    def __init__(self, individual, n_vars, nf, target_window, head_dropout=0):
        super().__init__()

        self.individual = individual
        self.n_vars = n_vars

        if self.individual:
            self.linears = nn.ModuleList()
            self.dropouts = nn.ModuleList()
            self.flattens = nn.ModuleList()
            for i in range(self.n_vars):
                self.flattens.append(nn.Flatten(start_dim=-2))
                self.linears.append(nn.Linear(nf, target_window))
                self.dropouts.append(nn.Dropout(head_dropout))
        else:
            self.flatten = nn.Flatten(start_dim=-2)
            self.linear = nn.Linear(nf, target_window)
            self.dropout = nn.Dropout(head_dropout)

    def forward(self, x_time, x_frequency):  # x: [bs x nvars x d_model x patch_num_out]
        x = torch.cat((x_time, x_frequency), dim=-2)  # [bs, nvars, 2 * d_model, patch_num_out]

        if self.individual:
            x_out = []
            for i in range(self.n_vars):
                z = self.flattens[i](x[:, i, :, :])  # z: [bs x 2 * d_model * patch_num]
                z = self.linears[i](z)  # z: [bs x target_window]
                z = self.dropouts[i](z)
                x_out.append(z)
            x = torch.stack(x_out, dim=1)  # x: [bs x nvars x target_window]
        else:
            x = self.flatten(x)
            x = self.linear(x)
            x = self.dropout(x)
        return x


class Model(nn.Module):
    def __init__(self, configs, max_seq_len: Optional[int] = 1024, d_k: Optional[int] = None, d_v: Optional[int] = None,
                 norm: str = 'BatchNorm', attn_dropout: float = 0.,
                 act: str = "gelu", key_padding_mask: bool = 'auto', padding_var: Optional[int] = None,
                 attn_mask: Optional[Tensor] = None, res_attention: bool = True,
                 pre_norm: bool = False, store_attn: bool = False, pe: str = 'zeros', learn_pe: bool = True,
                 pretrain_head: bool = False, head_type='flatten', verbose: bool = False, **kwargs):

        super().__init__()

        # load parameters
        bs = configs.batch_size
        c_in = configs.enc_in
        c_out = configs.c_out
        context_window = configs.seq_len
        target_window = configs.pred_len

        n_layers = configs.e_layers
        n_heads = configs.n_heads
        d_model = configs.d_model
        d_ff = configs.d_ff
        dropout = configs.dropout
        fc_dropout = configs.fc_dropout
        head_dropout = configs.head_dropout

        individual = configs.individual

        patch_len = configs.patch_len
        stride = configs.stride
        padding_patch = configs.padding_patch

        revin = configs.revin
        affine = configs.affine
        subtract_last = configs.subtract_last

        decomposition = configs.decomposition
        kernel_size = configs.kernel_size

        T_num_expert = configs.T_num_expert
        T_top_k = configs.T_top_k
        F_num_expert = configs.F_num_expert
        F_top_k = configs.F_top_k
        eta = configs.eta

        # RevIn
        self.revin = revin
        if self.revin: self.revin_layer = RevIN(c_in, affine=affine, subtract_last=subtract_last)

        # Patching
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch = padding_patch

        self.total_length = patch_len*2 + (int((target_window - patch_len)/stride + 1) - 1) * stride

        patch_num_in = int((context_window - patch_len) / stride + 1)
        patch_num_out = int((target_window - patch_len) / stride + 1)
        self.patch_num_out = patch_num_out
        self.target_window = target_window

        if padding_patch == 'end':  # can be modified to general case
            self.padding_patch_layer = nn.ReplicationPad1d((0, stride))
            patch_num_in += 1
            patch_num_out += 1

        print("patch_num_in, patch_num_out, patch_len", patch_num_in, patch_num_out, patch_len)

        # model
        self.decomposition = decomposition
        if self.decomposition:
            self.decomp_module = series_decomp(kernel_size)
            self.model_trend = PatchTST_MoE_cluster(bs=bs, c_in=c_in, c_out=c_out, context_window=context_window, target_window=target_window,
                                            patch_len=patch_len, stride=stride, patch_num_in=patch_num_in, patch_num_out=patch_num_out,
                                            max_seq_len=max_seq_len, n_layers=n_layers, d_model=d_model,
                                            n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm,
                                            attn_dropout=attn_dropout,
                                            dropout=dropout, act=act, key_padding_mask=key_padding_mask,
                                            padding_var=padding_var,
                                            attn_mask=attn_mask, res_attention=res_attention, pre_norm=pre_norm,
                                            store_attn=store_attn,
                                            pe=pe, learn_pe=learn_pe, fc_dropout=fc_dropout, head_dropout=head_dropout,
                                            padding_patch=padding_patch,
                                            pretrain_head=pretrain_head, head_type=head_type, individual=individual,
                                            revin=revin, affine=affine,
                                            subtract_last=subtract_last, verbose=verbose, T_num_expert=T_num_expert,
                                            T_top_k=T_top_k, F_num_expert=F_num_expert, F_top_k=F_top_k, eta=eta, **kwargs)
            self.model_res = PatchTST_MoE_cluster(bs=bs, c_in=c_in, c_out=c_out, context_window=context_window, target_window=target_window,
                                          patch_len=patch_len, stride=stride, patch_num_in=patch_num_in, patch_num_out=patch_num_out,
                                          max_seq_len=max_seq_len, n_layers=n_layers, d_model=d_model,
                                          n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm,
                                          attn_dropout=attn_dropout,
                                          dropout=dropout, act=act, key_padding_mask=key_padding_mask,
                                          padding_var=padding_var,
                                          attn_mask=attn_mask, res_attention=res_attention, pre_norm=pre_norm,
                                          store_attn=store_attn,
                                          pe=pe, learn_pe=learn_pe, fc_dropout=fc_dropout, head_dropout=head_dropout,
                                          padding_patch=padding_patch,
                                          pretrain_head=pretrain_head, head_type=head_type, individual=individual,
                                          revin=revin, affine=affine,
                                          subtract_last=subtract_last, verbose=verbose, T_num_expert=T_num_expert,
                                          T_top_k=T_top_k, F_num_expert=F_num_expert, F_top_k=F_top_k, eta=eta, **kwargs)
        else:
            self.model_time = PatchTST_MoE_cluster_time(bs=bs, c_in=c_in, c_out=c_out, context_window=context_window, target_window=target_window,
                                      patch_len=patch_len, stride=stride, patch_num_in=patch_num_in, patch_num_out=patch_num_out,
                                      max_seq_len=max_seq_len, n_layers=n_layers, d_model=d_model,
                                      n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm,
                                      attn_dropout=attn_dropout,
                                      dropout=dropout, act=act, key_padding_mask=key_padding_mask,
                                      padding_var=padding_var,
                                      attn_mask=attn_mask, res_attention=res_attention, pre_norm=pre_norm,
                                      store_attn=store_attn,
                                      pe=pe, learn_pe=learn_pe, fc_dropout=fc_dropout, head_dropout=head_dropout,
                                      padding_patch=padding_patch,
                                      pretrain_head=pretrain_head, head_type=head_type, individual=individual,
                                      revin=revin, affine=affine,
                                      subtract_last=subtract_last, verbose=verbose, T_num_expert=T_num_expert,
                                      T_top_k=T_top_k, F_num_expert=F_num_expert, F_top_k=F_top_k, eta=eta, **kwargs)
            self.model_frequency = PatchTST_MoE_cluster_frequency(bs=bs, c_in=c_in, c_out=c_out, context_window=context_window, target_window=target_window,
                                      patch_len=patch_len, stride=stride, patch_num_in=patch_num_in, patch_num_out=patch_num_out,
                                      max_seq_len=max_seq_len, n_layers=n_layers, d_model=d_model,
                                      n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm,
                                      attn_dropout=attn_dropout,
                                      dropout=dropout, act=act, key_padding_mask=key_padding_mask,
                                      padding_var=padding_var,
                                      attn_mask=attn_mask, res_attention=res_attention, pre_norm=pre_norm,
                                      store_attn=store_attn,
                                      pe=pe, learn_pe=learn_pe, fc_dropout=fc_dropout, head_dropout=head_dropout,
                                      padding_patch=padding_patch,
                                      pretrain_head=pretrain_head, head_type=head_type, individual=individual,
                                      revin=revin, affine=affine,
                                      subtract_last=subtract_last, verbose=verbose, T_num_expert=T_num_expert,
                                      T_top_k=T_top_k, F_num_expert=F_num_expert, F_top_k=F_top_k, eta=eta, **kwargs)
        # Head
        self.head_nf = 2 * d_model * patch_num_out
        self.n_vars = c_in
        self.individual = individual

        if head_type == 'flatten':
            self.head = Flatten_Head(self.individual, self.n_vars, self.head_nf, target_window,
                                     head_dropout=head_dropout)

    def forward(self, x):  # x: [Batch, Input length, Channel]
        # norm
        if self.revin:
            x = self.revin_layer(x, 'norm')
            x = x.permute(0, 2, 1)

        # do patching
        if self.padding_patch == 'end':
            x = self.padding_patch_layer(x)
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)  # z: [bs x nvars x patch_num_in x patch_len]
        x = x.permute(0, 1, 3, 2)  # z: [bs x nvars x patch_len x patch_num_in]

        # time-frequency-encoder
        s_time, h_time, x_time = self.model_time(x)  # x_time: [bs x nvars x d_model x patch_num_out]
        s_frequency, h_frequency, x_frequency = self.model_frequency(x)  # x_frequency: [bs x nvars x d_model x patch_num_out]

        output = self.head(x_time, x_frequency)  # z: [bs x nvars x target_window]

        # denorm
        if self.revin:
            output = output.permute(0, 2, 1)
            output = self.revin_layer(output, 'denorm')  # [bs, target_window, nvars]

        return s_time, s_frequency, output