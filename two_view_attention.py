# Cross-view transformers for multi-view analysis of unregistered medical images
# Copyright (C) 2021 Gijs van Tulder / Radboud University, the Netherlands
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
 
import math
import torch
import torch.nn as nn
from torch.cuda.amp import custom_fwd, custom_bwd
import tokenizer

class TwoViewAttentionModule(nn.Module):
    implementation = 'samplewise-directsum'

    def __init__(self, heads, features_a, features_b, embedding, downsampling=None,
                 compute_coeff_l1_loss=False,
                 tokens=None, token_layers=1, tokenize_a=False,
                 tied_embeddings=False):
        super().__init__()

        if tokens:
            # build tokenizer
            if tokenize_a:
                self.tokenizer_a = tokenizer.Tokenizer(features_a, tokens, token_layers)
            else:
                self.tokenizer_a = None
            self.tokenizer_b = tokenizer.Tokenizer(features_b, tokens, token_layers)
        else:
            # no tokenization, use pixels directly
            self.tokenizer_a = None
            self.tokenizer_b = None

        # embedding
        self.embed_a = nn.Conv1d(features_a, heads * embedding, kernel_size=1, bias=False)
        if tied_embeddings:
            self.embed_b = self.embed_a
        else:
            self.embed_b = nn.Conv1d(features_b, heads * embedding, kernel_size=1, bias=False)

        # heads combiner
        self.combine_heads = nn.Conv1d(features_b * heads, features_a, kernel_size=1, bias=False)

        # parameters
        self.heads = heads
        self.features_a = features_a
        self.features_b = features_b
        self.embedding = embedding
        self.downsampling = downsampling if downsampling != 1 else None
        self.compute_coeff_l1_loss = not (compute_coeff_l1_loss is False or compute_coeff_l1_loss is None)

    def downsample(self, z):
        if self.downsampling is not None:
            # downsample
            if z.ndim == 3:
                pool_fn = nn.functional.max_pool1d
            elif z.ndim == 4:
                pool_fn = nn.functional.max_pool2d
            elif z.ndim == 5:
                pool_fn = nn.functional.max_pool3d
            else:
                raise ValueError('Downsampling expects 1D, 2D or 3D input.')
            z = pool_fn(z, self.downsampling, ceil_mode=True)
        return z

    def upsample(self, s, orig_shape):
        if self.downsampling is not None:
            # upsample to original shape
            s = nn.functional.interpolate(s, scale_factor=self.downsampling)
            # crop to the correct size (max_pool may have added some padding)
            s = s[[slice(0, shp) for shp in orig_shape]]
        return s

    def forward(self, z_a, z_b, z_a_ds=None, z_b_ds=None):
        # downsample, unless downsampled z_a_ds and z_b_ds are given
        if z_a_ds is None:
            z_a_ds = self.downsample(z_a)
        if z_b_ds is None:
            z_b_ds = self.downsample(z_b)

        # tokenize, if required
        if self.tokenizer_a is not None:
            z_a_ds_shape = z_a_ds.shape
            z_a_ds, token_attn_a = self.tokenizer_a(z_a_ds)
        if self.tokenizer_b is not None:
            z_b_ds, _ = self.tokenizer_b(z_b_ds)

        # encode attention queries for A and keys for B
        # [batch, head * embedding, q1 * q2 * ...]
        q = self.embed_a(z_a_ds.flatten(2))
        # [batch, head * embedding, k1 * k2 * ...]
        k = self.embed_b(z_b_ds.flatten(2))

        if self.implementation == 'samplewise-directsum':
            # [batch, q, channel], [batch]
            s, l1 = MultiHeadAttentionDirectSum.apply(q.view(q.shape[0], self.heads, self.embedding, q.shape[2]),
                                                      k.view(k.shape[0], self.heads, self.embedding, k.shape[2]),
                                                      z_b_ds.flatten(2),
                                                      self.combine_heads.weight.view(self.heads, z_b_ds.shape[1], z_a_ds.shape[1]),
                                                      self.compute_coeff_l1_loss)
            # [batch, channel, q]
            s = s.permute(0, 2, 1)

        else:
            if self.implementation == 'samplewise':
                # [batch, channel, head, q], [batch]
                s, l1 = SamplewiseMultiHeadAttention.apply(q.view(q.shape[0], self.heads, self.embedding, q.shape[2]),
                                                           k.view(k.shape[0], self.heads, self.embedding, k.shape[2]),
                                                           z_b_ds.flatten(2), self.compute_coeff_l1_loss, 1)
            elif self.implementation == 'custom-gradient':
                # [batch, channel, head, q], [batch]
                s, l1 = MultiHeadAttention.apply(q.view(q.shape[0], self.heads, self.embedding, q.shape[2]),
                                                 k.view(k.shape[0], self.heads, self.embedding, k.shape[2]),
                                                 z_b_ds.flatten(2), self.compute_coeff_l1_loss)
            else:
                assert self.implementation == 'traditional'
                s, l1 = self.compute_attention(q, k, z_b_ds)

            # [batch, channel * head, q]
            s = s.view(s.shape[0], s.shape[1] * self.heads, -1)
            # [batch, channel, q]
            s = self.combine_heads(s)

        # [batch, channel, q1, q2, ...]
        s = s.view(s.shape[0], s.shape[1], *z_a_ds.shape[2:])

        # if required, go back from tokens to pixels
        if self.tokenizer_a is not None:
            s = self.tokenizer_a.reverse(s, token_attn_a).view(*z_a_ds_shape)

        if self.downsampling is not None:
            # upsample to z_a
            s = nn.functional.interpolate(s, scale_factor=self.downsampling)
            # crop to the correct size (max_pool may have added some padding)
            s = s[[slice(0, shp) for shp in z_a.shape]]

        if self.compute_coeff_l1_loss:
            return s, l1
        else:
            return s

    def compute_attention(self, q, k, v):
        # [batch, head, embedding, q]
        q = q.view(q.shape[0], self.heads, q.shape[1] // self.heads, q.shape[2])
        # [batch, head, embedding, k]
        k = k.view(k.shape[0], self.heads, k.shape[1] // self.heads, k.shape[2])

        # compute attention score for each pixel or voxel pair
        # - queries q: [batch, head, embedding, q1 * q2 * ...]
        # - keys k:    [batch, head, embedding, k1 * k2 * ...]

        # normalization for dot product attention
        norm = torch.tensor(1 / math.sqrt(k.shape[2]), device=q.device, dtype=q.dtype)

        # [batch, k, head, q]
        s = torch.einsum('bheq,,bhek->bkhq', q, norm, k)

        if self.compute_coeff_l1_loss:
            # compute l1 loss
            # [batch]
            l1 = torch.mean(torch.abs(s), dim=(1, 2, 3))

        # use softmax over all k
        s = torch.nn.functional.softmax(s, dim=1)

        # [batch, k, head * q]
        s = s.view(s.shape[0], s.shape[1], s.shape[2] * s.shape[3])

        # compute attention-weighted representations
        # [batch, channel, head * q]
        s = torch.bmm(v.flatten(2), s)

        if self.compute_coeff_l1_loss:
            return s, l1
        else:
            return s


class PostAttentionCombiner(nn.Module):
    # combines the features from the original view with the attention-based features from the other view
    # method "add":            features A + attn features from B
    # method "add-linear":     features A + linear(attn features from B)
    # method "layernorm(add-linear+dropout)":
    #                          layernorm(features A + dropout(linear(attn features from B)))
    # method "linear-linear":  linear_a(features A) + linear_b(attn features from B)
    # method "concatenate":    concatenate(features A, features B) on the feature dimension
    def __init__(self, ndim, features_src, features_attn=None, features_out=None, method='add'):
        super().__init__()
        features_attn = features_attn or features_src
        features_out = features_attn or features_out
        self.features_out = features_out
        self.method = method
        if self.method == 'add':
            assert features_src == features_attn
            assert features_src == features_out
        elif self.method == 'add-linear':
            assert features_src == features_out
            self.attn_linear = self.linear_map(ndim, features_attn, features_out)
        elif self.method == 'ln-add-linear-do':
            assert features_src == features_out
            self.attn_linear = self.linear_map(ndim, features_attn, features_out)
            self.attn_dropout = nn.Dropout()
            self.layernorm = LayerNormND(features_out)
        elif self.method == 'linear-linear':
            self.src_linear = self.linear_map(ndim, features_src, features_out)
            self.attn_linear = self.linear_map(ndim, features_attn, features_out)
        elif self.method == 'concatenate':
            self.features_out = features_src + features_attn
        else:
            raise ValueError('unknown combine function %s' % str(method))

    def forward(self, src, attn):
        if self.method == 'add':
            return src + attn
        elif self.method == 'add-linear':
            return src + self.attn_linear(attn)
        elif self.method == 'ln-add-linear-do':
            return self.layernorm(src + self.attn_dropout(self.attn_linear(attn)))
        elif self.method == 'linear-linear':
            # mapping separately and then adding is slightly more memory-efficient than concatenating
            return self.src_linear(src) + self.attn_linear(attn)
        elif self.method == 'concatenate':
            return torch.cat([src, attn], dim=1)
        else:
            raise ValueError('unknown combine function %s' % str(method))

    def linear_map(self, ndim, features_from, features_to):
        if ndim == 2:
            return nn.Linear(features_from, features_to, bias=False)
        elif ndim == 3:
            return nn.Conv1d(features_from, features_to, kernel_size=1, bias=False)
        elif ndim == 4:
            return nn.Conv2d(features_from, features_to, kernel_size=1, bias=False)
        elif ndim == 5:
            return nn.Conv3d(features_from, features_to, kernel_size=1, bias=False)
        else:
            raise ValueError('PostAttentionCombiner expects 0D, 1D, 2D or 3D input.')


class MultiHeadAttention(torch.autograd.Function):
    # Multi-head attention with custom gradients.
    # Computes the full coefficient matrix and recomputes it during backpropagation.
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, q, k, v):
        # compute attention score for each pixel or voxel pair
        # - queries q: [batch, head, embedding, q1 * q2 * ...]
        # - keys k:    [batch, head, embedding, k1 * k2 * ...]
        # - values v:  [batch, channel, k1 * k2 * ...]
        ctx.save_for_backward(q, k, v)

        # normalization for dot product attention coefficients
        norm = torch.tensor(1 / math.sqrt(k.shape[2]), device=q.device, dtype=q.dtype)

        # compute coefficients
        # [batch, k, head, q]
        coeff = torch.einsum('bheq,,bhek->bkhq', q, norm, k)

        # use softmax over all k
        coeff = coeff.softmax(dim=1)

        # compute attention-weighted representations
        # [batch, channel, head * q]
        output = torch.bmm(v, coeff.flatten(2))

        # [batch, channel, head, q]
        return output.view(output.shape[0], output.shape[1], q.shape[1], q.shape[3])

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output):
        # - grad_output: [batch, channel, head, q]
        # - queries q:   [batch, head, embedding, q1 * q2 * ...]
        # - keys k:      [batch, head, embedding, k1 * k2 * ...]
        # - values v:    [batch, channel, k1 * k2 * ...]
        q, k, v = ctx.saved_tensors

        # [batch, channel, head * q]
        grad_output = grad_output.flatten(2)

        # normalization for dot product attention coefficients
        norm = torch.tensor(1 / math.sqrt(k.shape[2]), device=q.device, dtype=q.dtype)

        # compute coefficients (already computed in forward pass, but not saved)
        # [batch, k, head * q]
        coeff_pre_softmax = torch.einsum('bheq,,bhek->bkhq', q, norm, k).flatten(2)

        # use softmax over all k
        coeff_post_softmax = coeff_pre_softmax.softmax(dim=1)

        # gradient for v
        # [batch, channel, k]
        grad_v = torch.bmm(grad_output, coeff_post_softmax.permute(0, 2, 1))

        # gradient for coeff post-softmax
        # [batch, k, head * q]
        grad_coeff = torch.bmm(v.permute(0, 2, 1), grad_output)

        # gradient for coeff pre-softmax
        # [batch, k, head * q]
        # use softmax
        grad_coeff = torch._softmax_backward_data(grad_coeff, coeff_post_softmax, 1, coeff_pre_softmax)

        # [batch, k, head, q]
        grad_coeff = grad_coeff.view(k.shape[0], k.shape[3], q.shape[1], q.shape[3])

        # gradient for q
        # [batch, head, embedding, q]
        grad_q = torch.einsum('bkhq,,bhek->bheq', grad_coeff, norm, k)

        # gradient for k
        # [batch, head, embedding, k]
        grad_k = torch.einsum('bkhq,,bheq->bhek', grad_coeff, norm, q)

        return grad_q, grad_k, grad_v, None, None


class SamplewiseMultiHeadAttention(torch.autograd.Function):
    # Multi-head attention with custom gradients.
    # Computes the sample-wise coefficient matrix and recomputes it during backpropagation.
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, q, k, v, mb_size=1):
        # compute attention score for each pixel or voxel pair
        # - queries q: [batch, head, embedding, q1 * q2 * ...]
        # - keys k:    [batch, head, embedding, k1 * k2 * ...]
        # - values v:  [batch, channel, k1 * k2 * ...]
        ctx.save_for_backward(q, k, v)
        ctx.mb_size = mb_size

        # normalization for dot product attention coefficients
        norm = torch.tensor(1.0 / math.sqrt(k.shape[2]), device=q.device, dtype=q.dtype)

        # create output
        # [block, channel, head * q]
        output = torch.empty((q.shape[0], v.shape[1], q.shape[1] * q.shape[3]),
                             dtype=v.dtype, device=v.device)

        # loop over blocks of samples
        for offset in range(0, q.shape[0], mb_size):
            end = min(offset + mb_size, q.shape[0])

            # compute coefficients
            # [batch, k, head, q]
            coeff = torch.einsum('bheq,,bhek->bkhq', q[offset:end], norm, k[offset:end])

            # use softmax over all k
            coeff = coeff.softmax(dim=1)

            # compute attention-weighted representations
            # [batch, channel, head * q]
            torch.bmm(v[offset:end], coeff.flatten(2), out=output[offset:end])

        # [batch, channel, head, q]
        return output.view(q.shape[0], v.shape[1], q.shape[1], q.shape[3])

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output):
        # - grad_output: [batch, channel, head, q]
        # - queries q:   [batch, head, embedding, q1 * q2 * ...]
        # - keys k:      [batch, head, embedding, k1 * k2 * ...]
        # - values v:    [batch, channel, k1 * k2 * ...]
        q, k, v = ctx.saved_tensors
        mb_size = ctx.mb_size

        # [batch, channel, head * q]
        grad_output = grad_output.flatten(2)

        # normalization for dot product attention coefficients
        norm = torch.tensor(1.0 / math.sqrt(k.shape[2]), device=q.device, dtype=q.dtype)

        # create outputs
        grad_q = torch.empty_like(q)
        grad_k = torch.empty_like(k)
        grad_v = torch.empty_like(v)

        # loop over blocks of samples
        for offset in range(0, q.shape[0], mb_size):
            end = min(offset + mb_size, q.shape[0])

            # compute coefficients (already computed in forward pass, but not saved)
            # [batch, k, head * q]
            coeff_pre_softmax = torch.einsum('bheq,,bhek->bkhq', q[offset:end], norm, k[offset:end]).flatten(2)

            # use softmax over all k
            coeff_post_softmax = torch.nn.functional.softmax(coeff_pre_softmax, dim=1)

            # gradient for v
            # [batch, channel, k]
            torch.bmm(grad_output[offset:end], coeff_post_softmax.permute(0, 2, 1), out=grad_v[offset:end])

            # gradient for coeff_post_softmax
            # [batch, k, head * q]
            grad_coeff_post_softmax = torch.bmm(v[offset:end].permute(0, 2, 1), grad_output[offset:end])

            # gradient for coeff_pre_softmax
            # [batch, k, head * q]
            grad_coeff_pre_softmax = torch._softmax_backward_data(grad_coeff_post_softmax, coeff_post_softmax, 1, coeff_pre_softmax)

            # [batch, k, head, q]
            grad_coeff_pre_softmax = grad_coeff_pre_softmax.view(coeff_pre_softmax.shape[0], k.shape[3], q.shape[1], q.shape[3])

            # gradient for q
            # [batch, head, embedding, q]
            grad_q[offset:end] = torch.einsum('bkhq,,bhek->bheq', grad_coeff_pre_softmax, norm, k[offset:end])

            # gradient for k
            # [batch, head, embedding, k]
            grad_k[offset:end] = torch.einsum('bkhq,,bheq->bhek', grad_coeff_pre_softmax, norm, q[offset:end])

        return grad_q, grad_k, grad_v, None


class MultiHeadAttentionDirectSum(torch.autograd.Function):
    # Multi-head attention with custom gradients and direct channel summing.
    # Computes the sample-wise coefficient matrix for each head separately.
    # Combines the (head * features) directly.
    # Most memory-efficient for large q, k and for large numbers of heads.
    # Repeats some computations during backpropagation.
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, q, k, v, w, compute_coeff_l1_loss, mb_size=1):
        # compute attention score for each pixel or voxel pair
        # - queries q: [batch, head, embedding, q1 * q2 * ...]
        # - keys k:    [batch, head, embedding, k1 * k2 * ...]
        # - values v:  [batch, channel_in, k1 * k2 * ...]
        # - weights w: [head, channel_in, channel_out]
        ctx.save_for_backward(q, k, v, w)
        ctx.compute_coeff_l1_loss = compute_coeff_l1_loss
        ctx.mb_size = mb_size

        # normalization for dot product attention coefficients
        norm = torch.tensor(1 / math.sqrt(k.shape[2]), device=q.device, dtype=q.dtype)

        # create output
        # [block, q, channel_out]
        output = torch.zeros((q.shape[0], q.shape[3], w.shape[2]), dtype=v.dtype, device=v.device)
        if compute_coeff_l1_loss:
            l1 = torch.zeros((q.shape[0],), dtype=v.dtype, device=v.device)

        # loop over heads
        for head in range(q.shape[1]):
            # loop over blocks of samples
            for offset in range(0, q.shape[0], mb_size):
                end = min(offset + ctx.mb_size, q.shape[0])

                # compute coefficients
                # [batch, q, k] <- [batch, q, embedding] * [batch, embedding, k]
                coeff_pre_softmax = torch.bmm(q[offset:end, head].permute(0, 2, 1) * norm, k[offset:end, head])

                # use softmax over all k
                # [batch, q, k]
                coeff_post_softmax = coeff_pre_softmax.softmax(dim=2)

                if compute_coeff_l1_loss:
                    # compute l1 loss
                    # [batch]
                    l1[offset:end] += torch.mean(coeff_pre_softmax.abs_(), dim=(1, 2))

                # release memory
                del coeff_pre_softmax

                # compute attention-weighted representations
                # [batch, q, channel_in] <- [batch, q, k] * [batch, k, channel_in]
                feat = torch.bmm(coeff_post_softmax, v[offset:end].permute(0, 2, 1))

                # compute weighted feature combination
                # [batch * q, channel_out] <- [batch * q, channel_in] * [channel_in, channel_out]
                torch.addmm(output[offset:end].view(-1, output.shape[2]),
                            feat.view(-1, feat.shape[2]), w[head],
                            out=(output[offset:end].view(-1, output.shape[2])))

        if compute_coeff_l1_loss:
            # compute mean over heads
            l1 /= q.shape[1]

            # [batch, q, channel_out], [batch]
            return output, l1
        else:
            # [batch, q, channel_out]
            return output

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output, grad_l1=None):
        # - grad_output: [batch, q, channel_out]
        # - grad l1:     [batch]
        # - queries q:   [batch, head, embedding, q1 * q2 * ...]
        # - keys k:      [batch, head, embedding, k1 * k2 * ...]
        # - values v:    [batch, channel_in, k1 * k2 * ...]
        # - weights w:   [head, channel_in, channel_out]
        q, k, v, w = ctx.saved_tensors
        compute_coeff_l1_loss = ctx.compute_coeff_l1_loss
        mb_size = ctx.mb_size

        # normalization for dot product attention coefficients
        norm = torch.tensor(1 / math.sqrt(k.shape[2]), device=q.device, dtype=q.dtype)

        # create outputs
        grad_q = torch.empty_like(q)
        grad_k = torch.empty_like(k)
        grad_v = torch.zeros_like(v)
        grad_w = torch.zeros_like(w)

        if compute_coeff_l1_loss:
            # l1 is the mean over heads and coefficients
            grad_l1 = grad_l1 / (q.shape[1] * q.shape[3] * k.shape[3])

        # loop over heads
        for head in range(q.shape[1]):
            # loop over blocks of samples
            for offset in range(0, q.shape[0], mb_size):
                end = min(offset + ctx.mb_size, q.shape[0])

                # compute coefficients (already computed in forward pass, but not saved)
                # [batch, q, k] <- [batch, q, embedding] * [batch, embedding, k]
                coeff_pre_softmax = torch.bmm(q[offset:end, head].permute(0, 2, 1) * norm, k[offset:end, head])

                # use softmax over all k
                # [batch, q, k]
                # use softmax over all k
                coeff_post_softmax = coeff_pre_softmax.softmax(dim=2)

                # compute attention-weighted representations
                # [batch, q, channel_in] <- [batch, q, k] * [batch, k, channel_in]
                feat = torch.bmm(coeff_post_softmax, v[offset:end].permute(0, 2, 1))

                # gradient for w
                # [channel_in, channel_out] <- [batch, q, channel_in] * [batch, q, channel_out]
                grad_w[head] += torch.einsum('bqi,bqo->io', feat, grad_output[offset:end])

                # gradient for feat
                # [batch, q, channel_in] <- [channel_in, channel_out] * [batch, q, channel_out]
                grad_feat = torch.einsum('io,bqo->bqi', w[head], grad_output[offset:end])

                # gradient for v
                # [batch, channel_in, k] <- [batch, channel_in, q] * [batch, q, k]
                grad_v[offset:end] += torch.bmm(grad_feat.permute(0, 2, 1), coeff_post_softmax)

                # gradient for coeff post-softmax
                # [batch, q, k] <- [batch, q, channel_in] * [batch, channel_in, k]
                grad_coeff = torch.bmm(grad_feat, v[offset:end])

                # gradient for coeff pre-softmax
                # [batch, q, k]
                # use softmax
                grad_coeff = torch._softmax_backward_data(grad_coeff, coeff_post_softmax, 2, coeff_pre_softmax)

                if compute_coeff_l1_loss:
                    # add gradient from L1 loss
                    # (grad_l1 is already divided by the number of elements to represent the mean)
                    # [batch, q, k] <- [batch, q, k] * [batch, None, None]
                    grad_coeff.addcmul_(coeff_pre_softmax.sign_(), grad_l1[offset:end, None, None])

                # release memory
                del coeff_pre_softmax

                # gradient for q
                # [batch, embedding, q] <- [batch, embedding, k] * [batch, k, q]
                torch.bmm(k[offset:end, head] * norm, grad_coeff.permute(0, 2, 1), out=grad_q[offset:end, head])

                # gradient for k
                # [batch, embedding, k] <- [batch, embedding, q] * [batch, q, k]
                torch.bmm(q[offset:end, head] * norm, grad_coeff, out=grad_k[offset:end, head])

        return grad_q, grad_k, grad_v, grad_w, None, None, None


class LayerNormND(nn.Module):
    # apply LayerNorm to the final dimension
    def __init__(self, hidden_size, eps=1e-12):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        shp = x.shape
        x = x.view(shp[0], shp[1], -1)
        u = x.mean(2, keepdim=True)
        s = (x - u).pow(2).mean(2, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        x = self.weight[:, None] * x + self.bias[:, None]
        return x.view(*shp)


class DummyContext():
    def save_for_backward(self, *args):
        self.saved_tensors = args


if __name__ == '__main__':
    import timeit

    batch = 3
    head = 5
    embedding = 7
    q1 = 5
    k1 = 7
    channel = 4

#   batch = 3
#   head = 2
#   embedding = 2
#   q1 = 2
#   k1 = 2
#   channel = 2

    device = 'cpu'
    q = torch.randn([batch, head, embedding, q1], dtype=torch.double, requires_grad=True, device=device)
    k = torch.randn([batch, head, embedding, k1], dtype=torch.double, requires_grad=True, device=device)
    v = torch.randn([batch, channel, k1], dtype=torch.double, requires_grad=True, device=device)
    w_random = torch.randn([head, channel, channel], dtype=torch.double, requires_grad=True, device=device)
    w_eye = torch.eye(channel, dtype=torch.double, requires_grad=True, device=device).repeat(head, 1, 1)

    print('SampleWiseMultiHeadAttention')
    SamplewiseMultiHeadAttention.apply(q, k, v)
    print('MultiHeadAttention')
    MultiHeadAttention.apply(q, k, v)
    print('MultiHeadAttentionDirectSum')
    MultiHeadAttentionDirectSum.apply(q, k, v, w_random, True)

    print(torch.allclose(MultiHeadAttention.apply(q, k, v).sum(dim=2).permute(0, 2, 1),
                         MultiHeadAttentionDirectSum.apply(q, k, v, w_eye, False)))
    print(torch.allclose(MultiHeadAttention.apply(q, k, v),
                         SamplewiseMultiHeadAttention.apply(q, k, v)))

    result = torch.autograd.gradcheck(MultiHeadAttentionDirectSum.apply,
                  (q, k, v, w_random, True), eps=1e-6, atol=1e-4)
    print('gradcheck MultiHeadAttentionDirectSum', result)

    result = torch.autograd.gradcheck(MultiHeadAttention.apply,
                  (q, k, v), eps=1e-6, atol=1e-4)
    print('gradcheck MultiHeadAttention', result)

    result = torch.autograd.gradcheck(SamplewiseMultiHeadAttention.apply,
                                      (q, k, v), eps=1e-6, atol=1e-4)
    print('gradcheck SamplewiseMultiHeadAttention', result)
