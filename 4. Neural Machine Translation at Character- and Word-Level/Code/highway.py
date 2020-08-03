#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2019-20: Homework 5
"""

import torch
import torch.nn as nn

class Highway(nn.Module):
    # Remember to delete the above 'pass' after your implementation
    ### YOUR CODE HERE for part 1f

    def __init__(self, e_word: int):
        """
        Instantiates two learnable linear projections Wproj, Wgate of sizes e_word*e_word each.

        @param e_word (int): final word embedding length.
        """
        
        super(Highway, self).__init__()

        self.Wproj = nn.Linear(e_word, e_word, bias=True)
        self.Wgate = nn.Linear(e_word, e_word, bias=True)

        self.x_highway = None


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies a skip-connection controlled by a dynamic gate to transform x_conv_out to final word embeddings.

        @param x (Tensor, size = batch_size * e_word): batch of word vectors generated from the convolution network (x_conv_out in document).

        @returns x_word_emb (Tensor, size = batch_size * e_word): final word embeddings for each word in batch.
        """

        mat_size = x.size()

        x_proj = self.Wproj(x).clamp(min=0)
        assert x_proj.size() == mat_size, "x_proj is incorrect size; expected {} but got {}". \
            format(mat_size, tuple(x_proj.size()))
        # print("\nx_proj =\n", x_proj)

        x_gate = torch.sigmoid(self.Wgate(x))
        assert x_gate.size() == mat_size, "x_gate is incorrect size; expected {} but got {}". \
            format(mat_size, tuple(x_gate.size()))
        # print("\nx_gate =\n", x_gate)

        self.x_highway = x_gate * x_proj + (1 - x_gate) * x
        assert self.x_highway.size() == mat_size, "x_highway is incorrect size; expected {} but got {}". \
            format(mat_size, tuple(self.x_highway.size()))
        # print("\nx_highway =\n", self.x_highway)

        return self.x_highway


    ### END YOUR CODE

