import torch
import numpy as np
from torch import nn
from inspect import isfunction
import torch.nn.functional as F


def exists(val):
    return val is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


# feedforward
class GEGLU(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


class FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=0.):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)
        project_in = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU()
        ) if not glu else GEGLU(dim, inner_dim)

        self.net = nn.Sequential(
            project_in,
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim_out)
        )

    def forward(self, x):
        return self.net(x)


class SelfAttention(nn.Module):
    def __init__(self, query_dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(query_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim), nn.Dropout(dropout))

    def forward(self, x, instance_options={}):
        grounding_input = instance_options.get('grounding_input', None)
        drop_box_mask = instance_options.get('drop_box_mask', None)
        q = self.to_q(x)  # B*N*(H*C)
        k = self.to_k(x)  # B*N*(H*C)
        v = self.to_v(x)  # B*N*(H*C)

        B, N, HC = q.shape
        H = self.heads
        C = HC // H

        q = q.view(B, N, H, C).permute(
            0, 2, 1, 3).reshape(B * H, N, C)  # (B*H)*N*C
        k = k.view(B, N, H, C).permute(
            0, 2, 1, 3).reshape(B * H, N, C)  # (B*H)*N*C
        v = v.view(B, N, H, C).permute(
            0, 2, 1, 3).reshape(B * H, N, C)  # (B*H)*N*C

        sim = torch.einsum('b i c, b j c -> b i j', q, k) * \
            self.scale  # (B*H)*N*N

        if grounding_input is not None:
            # att_masks: B*(n_objs*4)*sqrt(w_h)*sqrt(w_h)
            n_objs = grounding_input["att_masks"].shape[1]

            att_masks_reshape = None
            if N - n_objs * 4 == 64 * 64:
                att_masks_reshape = grounding_input["att_masks"]
                att_key = "att_masks_selfAtt64"

            # if the mask is all zeros or box&mask are dropped, we do not need to mask the self-attention
            if att_masks_reshape is not None and torch.sum(att_masks_reshape) > 0.0 and not drop_box_mask:
                # the attention mask is shared across all blocks for more efficient forwarding
                # we only need to get the mask for the first block.
                if att_key in grounding_input:
                    att_masks_ = grounding_input[att_key]
                else:
                    w_h = att_masks_reshape.shape[2] * \
                        att_masks_reshape.shape[3]
                    att_masks_ = torch.ones(B, 1, N, N).type(
                        att_masks_reshape.dtype).to(att_masks_reshape.device)
                    # mask the self-attention between visual tokens
                    # all object patches do not exchange information with each other
                    # but object patches share information with all background regions
                    att_masks_reshape_v = att_masks_reshape.view(
                        B * n_objs, w_h, 1)

                    # get the self-attention between att_masks_reshape with batch matrix multiplication
                    #  1. att_masks_reshape_v: (B*n_objs)*w_h*1
                    #  2. att_masks_reshape_v.permute(0,1,3,2): (B*n_objs)*1*w_h
                    #  3. torch.bmm(att_masks_reshape_v, att_masks_reshape_v.permute(0,2,1)): (B*n_objs)*w_h*w_h
                    self_att_all = torch.bmm(
                        att_masks_reshape_v, att_masks_reshape_v.permute(0, 2, 1))

                    # sum the values alone the n_objs dimension
                    #  1. self_att_all: B*n_objs*w_h*w_h
                    #  2. self_att_all.sum(dim=1): B*w_h*w_h
                    self_att_ind_objs = self_att_all.view(
                        B, n_objs, w_h, w_h).sum(dim=1)

                    # sum att_masks_reshape alone the n_objs dimension and measure the self-attention with batch matrix multiplication
                    #  1. att_masks_reshape_all: B*n_objs*w_h*1
                    #  2. att_masks_reshape_all.sum(dim=1): B*w_h*1
                    #  3. set non-zero values to 1: B*w_h*1
                    #  4. use batch matrix multiplication to measure the self-attention: B*w_h*w_h
                    att_masks_reshape_all = att_masks_reshape_v.view(
                        B, n_objs, w_h, 1).sum(dim=1)
                    att_masks_reshape_all[att_masks_reshape_all >= 1.0] = 1.0
                    self_att_all_objs = torch.bmm(
                        att_masks_reshape_all, att_masks_reshape_all.permute(0, 2, 1))

                    # get the masks for avoiding information leakage between object patches
                    visual_token_masks = self_att_all_objs + self_att_ind_objs

                    # avoid the communications between objects and background
                    # objects-background can not communicate
                    visual_token_masks[self_att_ind_objs < 1.0] = 0.0
                    visual_token_masks[self_att_ind_objs >=
                                       1.0] = 1.0  # binay mask

                    att_masks_[:, :, :w_h, :w_h] = visual_token_masks.view(
                        B, 1, w_h, w_h)

                    # mask the self-attention between grounded tokens and the visual tokens
                    att_masks_reshape = att_masks_reshape.view(
                        B, 1, n_objs, w_h)
                    # the order of inputs are [box, point, scribble, mask]. only box and mask need to have masked self-attention
                    att_masks_[:, :, w_h:, :w_h] = att_masks_reshape.repeat(
                        1, 1, 4, 1)
                    att_masks_[:, :, w_h + n_objs:w_h + n_objs * 3, :w_h] = 1
                    att_masks_[:, :, :w_h, w_h:] = att_masks_reshape.permute(
                        0, 1, 3, 2).repeat(1, 1, 1, 4)
                    att_masks_[:, :, :w_h, w_h + n_objs:w_h + n_objs * 3] = 1

                    # add 1e-9 along the diagonal to avoid nan
                    diagonal_epsilon = torch.eye(N).view(1, 1, N, N).type(
                        att_masks_.dtype).to(att_masks_.device) * 1e-9
                    att_masks_ = att_masks_ + diagonal_epsilon

                    # save the masks for later blocks
                    grounding_input['next_att_mask'] = att_masks_
                # the larger the threshold is, the smaller area the grounded info can influence
                # if threshold = 1.01, all grounded info has no influence on the rest of the patches.
                sim = sim.view(B, H, N, N).masked_fill(
                    att_masks_ <= 0.0, -np.inf).view(B * H, N, N)  # -np.inf

        attn = sim.softmax(dim=-1)  # (B*H)*N*N

        out = torch.einsum('b i j, b j c -> b i c', attn, v)  # (B*H)*N*C
        out = out.view(B, H, N, C).permute(
            0, 2, 1, 3).reshape(B, N, (H * C))  # B*N*(H*C)
        return self.to_out(out)


class GatedSelfAttentionDense(nn.Module):
    def __init__(self, query_dim, context_dim, n_heads, d_head):
        super().__init__()
        # we need a linear projection since we need cat visual feature and obj feature
        self.linear = nn.Linear(context_dim, query_dim)

        self.attn = SelfAttention(
            query_dim=query_dim, heads=n_heads, dim_head=d_head)
        self.ff = FeedForward(query_dim, glu=True)

        self.norm1 = nn.LayerNorm(query_dim)
        self.norm2 = nn.LayerNorm(query_dim)

        self.register_parameter('alpha_attn', nn.Parameter(torch.tensor(0.)))
        self.register_parameter('alpha_dense', nn.Parameter(torch.tensor(0.)))

        # this can be useful: we can externally change magnitude of tanh(alpha)
        # for example, when it is set to 0, then the entire model is same as original one
        self.scale = 1

    def forward(self, x, instance_options={}):
        objs = instance_options['objs']

        N_visual = x.shape[1]
        objs = self.linear(objs)
        attention_output = self.attn(self.norm1(
            torch.cat([x, objs], dim=1)), instance_options=instance_options)
        x = x + self.scale * \
            torch.tanh(self.alpha_attn) * attention_output[:, 0:N_visual, :]
        x = x + self.scale * \
            torch.tanh(self.alpha_dense) * self.ff(self.norm2(x))

        return x.to(torch.float16)
