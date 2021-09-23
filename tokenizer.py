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

class Tokenizer(nn.Module):
    # tokens similar to the tokens in the Visual Transformers paper
    def __init__(self, channels, tokens=16, layers=1):
        super().__init__()

        # first tokenization layer
        self.attn = nn.Conv1d(channels, tokens, kernel_size=1, bias=False)

        # W_T-to-R mapping tokens from the previous layer to the current layer
        self.token_maps = nn.ModuleList(
            [nn.Linear(channels, channels, bias=False) for l in range(layers - 1)]
        )

    def forward(self, x):
        # x: [batch, channel, h, w, ...]

        # first tokenization layer
        # compute attention weights
        # [batch, token, h * w * ...]
        attn = self.attn(x.flatten(2))
        # [batch, h * w * ..., token]
        attn = nn.functional.softmax(attn.permute(0, 2, 1), dim=1)

        # compute attention-weighted tokens
        # [batch, channel, token] <- [batch, channel, h * w * ...] * [batch, h * w * ..., token]
        tokens = torch.bmm(x.flatten(2), attn)

        # subsequent recurrent tokenizer layers
        for token_map in self.token_maps:
            # map tokens
            # [batch, channel, token]
            tokens = token_map(tokens.permute(0, 2, 1)).permute(0, 2, 1)
            # compute attention weights
            # [batch, token, h * w] <- [batch, token, channel] * [batch, channel, h * w]
            attn = torch.bmm(tokens.permute(0, 2, 1), x.flatten(2))
            # [batch, h * w, token]
            attn = nn.functional.softmax(attn.permute(0, 2, 1), dim=1)
            # recompute tokens
            # [batch, channel, token] <- [batch, channel, h * w] * [batch, h * w, token]
            tokens = torch.bmm(x.flatten(2), attn)

        return tokens, attn

    def reverse(self, y, attn):
        # y: [batch, channel, token]
        # attn: [batch, h * w * ..., token]
        # return: [batch, channel, h * w * h]
        return torch.bmm(y, attn.permute(0, 2, 1))
