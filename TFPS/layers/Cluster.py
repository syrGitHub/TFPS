import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from layers.Constraint import D_constraint1, D_constraint2

class EDESC(nn.Module):

    def __init__(self,
                 d_model,
                 n_clusters,
                 eta,
                 c_out,
                 bs,
                 patch_len,
                 stride):
        super(EDESC, self).__init__()
        self.n_clusters = n_clusters
        self.eta = eta
        self.c_out = c_out
        self.bs = bs
        self.patch_len = patch_len
        self.stride = stride

        # Subspace bases proxy
        # self.D = Parameter(torch.Tensor(n_z, n_clusters))
        n_z = c_out * d_model
        self.d = int(n_z / n_clusters)
        self.D = Parameter(torch.Tensor(n_clusters * self.d, n_clusters * self.d))

    def reverse_unfold(self, z, original_length, stride):
        # z: [bs x patch_num x nvars x patch_len]
        bs, patch_num, nvars, patch_len = z.size()
        output = torch.zeros((bs, nvars, original_length), device=z.device)
        patch_counts = torch.zeros((bs, nvars, original_length), device=z.device)

        for i in range(patch_num):
            start = i * stride
            end = start + patch_len
            if end > original_length:
                end = original_length

            # Dynamically adjust the patch length if it exceeds the original length
            current_patch_len = end - start

            output[:, :, start:end] += z[:, i, :, :current_patch_len]
            patch_counts[:, :, start:end] += 1

        output /= patch_counts
        output = torch.reshape(output, (output.shape[0], output.shape[2], output.shape[1]))
        return output   # output: [bs, c_out, context_window]

    def forward(self, z):  # z: [bs * patch_num_out x nvars * d_model]
        # x_bar, z = self.ae(x)  # x_bar: [bs * patch_num x nvars * patch_len]
        # x_bar = torch.reshape(x_bar, (self.bs, -1, self.c_out, self.patch_len))
        # x_bar = self.reverse_unfold(x_bar, length, self.stride)    # x_bar: [bs, c_out, context_window]
        s = None
        
        # Calculate subspace affinity
        for i in range(self.n_clusters):

            si = torch.sum(torch.pow(torch.mm(z, self.D[:, i * self.d:(i + 1) * self.d]), 2), 1, keepdim=True)
            if s is None:
                s = si
            else:
                s = torch.cat((s, si), 1)
        s = (s + self.eta * self.d) / ((self.eta + 1) * self.d)
        s = (s.t() / torch.sum(s, 1)).t()  # Eq 13  [b_s, num_expert]
        return s, z

    def total_loss(self, pred, target, dim, n_clusters, beta): # x, x_bar,
        # Reconstruction loss   Eq 9
        # reconstr_loss = F.mse_loss(x_bar, x)

        # Subspace clustering loss  Eq 15
        kl_loss = F.kl_div(pred.log(), target)

        # Constraints   Eq 12
        d_cons1 = D_constraint1()
        d_cons2 = D_constraint2()
        loss_d1 = d_cons1(self.D)
        loss_d2 = d_cons2(self.D, dim, n_clusters)

        # Total_loss    Eq 16
        total_loss = beta * kl_loss + loss_d1 + loss_d2  # reconstr_loss +

        return total_loss
