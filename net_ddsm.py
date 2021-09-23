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
 
import torch
import torch.nn as nn
import torchvision.models

import two_view_attention

from net import register_model

@register_model
class SingleViewResNet18ShallowTop(nn.Module):
    def __init__(self, in_channels, outputs, pretrained=True, dropout=None):
        super().__init__()

        assert in_channels == 1, 'in_channels expected to be 1'
        self.in_channels = in_channels
        self.outputs = outputs
        self.dropout = dropout

        # copy ResNet layers
        resnet = torchvision.models.resnet18(pretrained=pretrained)
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.avgpool = resnet.avgpool

        self.fcn = nn.Sequential(
            nn.Dropout(self.dropout) if self.dropout else nn.Identity(),
            nn.Linear(512, outputs),
        )

    def forward(self, x):
        # from 1 channel to 3
        x = x.expand(-1, 3, -1, -1)

        # resnet convolution
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # resnet pooling
        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        # final top layers
        return self.fcn(x)


@register_model
class LateJoinResNet18ShallowTop(nn.Module):
    def __init__(self, in_channels, outputs, view_dropout=False, pretrained=True, dropout=None):
        super().__init__()

        assert in_channels == 1, 'in_channels expected to be 1'
        self.in_channels = in_channels
        self.outputs = outputs
        self.view_dropout = view_dropout
        self.dropout = dropout

        # copy the first part of a ResNet18
        pre_a = SingleViewResNet18ShallowTop(in_channels, 1, pretrained=pretrained)
        pre_b = SingleViewResNet18ShallowTop(in_channels, 1, pretrained=pretrained)
        self.pre_a = nn.Sequential(
            pre_a.conv1, pre_a.bn1, pre_a.relu, pre_a.maxpool,
            pre_a.layer1, pre_a.layer2, pre_a.layer3, pre_a.layer4,
            pre_a.avgpool, nn.Flatten())
        self.pre_b = nn.Sequential(
            pre_b.conv1, pre_b.bn1, pre_b.relu, pre_b.maxpool,
            pre_b.layer1, pre_b.layer2, pre_b.layer3, pre_b.layer4,
            pre_b.avgpool, nn.Flatten())

        # final part
        self.fcn = nn.Sequential(
            nn.Dropout(self.dropout) if self.dropout else nn.Identity(),
            nn.Linear(2 * 512, self.outputs),
        )

    def forward(self, x_a, x_b):
        # from 1 channel to 3
        x_a = x_a.expand(-1, 3, -1, -1)
        x_b = x_b.expand(-1, 3, -1, -1)

        # preliminary convolutions
        z_a = self.pre_a(x_a)
        z_b = self.pre_b(x_b)

        # which views to use?
        if self.view_dropout:
            z = cat_with_view_dropout(self.training, z_a, z_b)
        else:
            # concatenate views
            z = torch.cat([z_a, z_b], dim=1)

        return self.fcn(z)


@register_model
class TwoViewAttentionResNet18ShallowTop(nn.Module):
    def __init__(self, in_channels, outputs, pretrained=True,
                 heads=16, attention_downsampling=4, attention_combine='add',
                 attention_bidirectional=False, attention_l1_loss=False,
                 attention_tokens=None, attention_token_layers=1, attention_tokenize_a=False,
                 dropout=None, view_dropout=False):
        super().__init__()

        assert in_channels == 1, 'in_channels expected to be 1'
        self.in_channels = in_channels
        self.outputs = outputs
        self.heads = heads
        self.attention_downsampling = attention_downsampling
        self.attention_combine = attention_combine
        self.attention_bidirectional = attention_bidirectional
        self.attention_l1_loss = attention_l1_loss
        self.attention_tokens = attention_tokens
        self.attention_token_layers = attention_token_layers
        self.attention_tokenize_a = attention_tokenize_a
        self.pretrained = pretrained
        self.dropout = dropout
        self.view_dropout = view_dropout

        # copy the first part of a ResNet18
        pre_a = SingleViewResNet18ShallowTop(in_channels, 1, pretrained=pretrained)
        pre_b = SingleViewResNet18ShallowTop(in_channels, 1, pretrained=pretrained)
        self.pre_a = nn.Sequential(
            pre_a.conv1, pre_a.bn1, pre_a.relu, pre_a.maxpool,
            pre_a.layer1, pre_a.layer2, pre_a.layer3)
        self.post_a = nn.Sequential(
            pre_a.layer4, pre_a.avgpool, nn.Flatten())
        self.pre_b = nn.Sequential(
            pre_b.conv1, pre_b.bn1, pre_b.relu, pre_b.maxpool,
            pre_b.layer1, pre_b.layer2, pre_b.layer3)
        self.post_b = nn.Sequential(
            pre_b.layer4, pre_b.avgpool, nn.Flatten())

        # final part
        self.fcn = nn.Sequential(
            nn.Linear(2 * 512, self.outputs),
        )

        # attention module from B to A
        self.attn_b_to_a = two_view_attention.TwoViewAttentionModule(heads, downsampling=attention_downsampling,
                                                                     compute_coeff_l1_loss=attention_l1_loss,
                                                                     tokens=self.attention_tokens,
                                                                     token_layers=self.attention_token_layers,
                                                                     tokenize_a=self.attention_tokenize_a,
                                                                     features_a=256, features_b=256, embedding=32)
        self.attn_combiner_b_to_a = two_view_attention.PostAttentionCombiner(4, 256, method=self.attention_combine)

        if self.attention_bidirectional:
            # attention module from A to B
            self.attn_a_to_b = two_view_attention.TwoViewAttentionModule(heads, downsampling=attention_downsampling,
                                                                         compute_coeff_l1_loss=attention_l1_loss,
                                                                         tokens=self.attention_tokens,
                                                                         token_layers=self.attention_token_layers,
                                                                         tokenize_a=self.attention_tokenize_a,
                                                                         features_a=256, features_b=256, embedding=32)
            self.attn_combiner_a_to_b = two_view_attention.PostAttentionCombiner(4, 256, method=self.attention_combine)

    def forward(self, x_a, x_b):
        # from 1 channel to 3
        x_a = x_a.expand(-1, 3, -1, -1)
        x_b = x_b.expand(-1, 3, -1, -1)

        # preliminary convolutions
        z_a = self.pre_a(x_a)
        z_b = self.pre_b(x_b)

        # downsample for attention (both directions use the same factor)
        z_a_ds = self.attn_b_to_a.downsample(z_a)
        z_b_ds = self.attn_b_to_a.downsample(z_b)

        # dropout?
        if self.dropout:
            z_a_ds = nn.functional.dropout2d(z_a_ds, self.dropout, self.training)
            z_b_ds = nn.functional.dropout2d(z_b_ds, self.dropout, self.training)

        # get attention-based features from B to A
        s_b_to_a, *extra_b_to_a = self.attn_b_to_a(z_a, z_b, z_a_ds, z_b_ds)

        # dropout?
        if self.dropout:
            s_b_to_a = nn.functional.dropout2d(s_b_to_a, self.dropout, self.training)

        # combine A with attention from B to A
        z_a = self.attn_combiner_b_to_a(z_a, s_b_to_a)

        # also from A to B?
        extra_a_to_b = []
        if self.attention_bidirectional:
            # get attention-based features from A to B
            s_a_to_b, *extra_a_to_b = self.attn_a_to_b(z_b, z_a, z_b_ds, z_a_ds)

            # dropout?
            if self.dropout:
                s_b_to_a = nn.functional.dropout2d(s_b_to_a, self.dropout, self.training)

            # combine A with attention from B to A
            z_b = self.attn_combiner_a_to_b(z_b, s_a_to_b)

        # post-attention convolution
        z_a = self.post_a(z_a)
        z_b = self.post_b(z_b)

        # which views to use?
        if self.view_dropout:
            z = cat_with_view_dropout(self.training, z_a, z_b)
        else:
            # concatenate views
            z = torch.cat([z_a, z_b], dim=1)

        # concatenate views for the FCN
        y = self.fcn(z)

        # return with attention l1 loss if required
        return (y, *extra_b_to_a, *extra_a_to_b)


@register_model
class TwoViewAttentionLevel2ResNet18ShallowTop(TwoViewAttentionResNet18ShallowTop):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # move the ResNet layer3 block to the post-attention layer
        self.pre_a, layer3_a = self.pre_a[:-1], self.pre_a[-1]
        self.pre_b, layer3_b = self.pre_b[:-1], self.pre_b[-1]
        self.post_a = nn.Sequential(layer3_a, *self.post_a)
        self.post_b = nn.Sequential(layer3_b, *self.post_b)

        # attention module from B to A
        self.attn_b_to_a = two_view_attention.TwoViewAttentionModule(self.heads, downsampling=self.attention_downsampling,
                                                                     compute_coeff_l1_loss=self.attention_l1_loss,
                                                                     tokens=self.attention_tokens,
                                                                     token_layers=self.attention_token_layers,
                                                                     tokenize_a=self.attention_tokenize_a,
                                                                     features_a=128, features_b=128, embedding=32)
        self.attn_combiner_b_to_a = two_view_attention.PostAttentionCombiner(4, 128, method=self.attention_combine)

        if self.attention_bidirectional:
            # attention module from A to B
            self.attn_a_to_b = two_view_attention.TwoViewAttentionModule(self.heads, downsampling=self.attention_downsampling,
                                                                         compute_coeff_l1_loss=self.attention_l1_loss,
                                                                         tokens=self.attention_tokens,
                                                                         token_layers=self.attention_token_layers,
                                                                         tokenize_a=self.attention_tokenize_a,
                                                                         features_a=128, features_b=128, embedding=32)
            self.attn_combiner_a_to_b = two_view_attention.PostAttentionCombiner(4, 128, method=self.attention_combine)



def cat_with_view_dropout(training, a, b):
    if training:
        # 0: a, 1: a+b, 2: b
        r = torch.randint(3, size=(a.shape[0],))
        # use weight 2 for single views, weight 1 if both views are used
        w_a = (2 * (r == 0) + 1 * (r == 1)).to(device=a.device, dtype=a.dtype)
        w_b = (2 * (r == 2) + 1 * (r == 1)).to(device=b.device, dtype=b.dtype)
        w_a = w_a[:, None].expand(*a.shape)
        w_b = w_b[:, None].expand(*b.shape)
        a *= w_a
        b *= w_b

    # concatenate views
    return torch.cat([a, b], dim=1)
