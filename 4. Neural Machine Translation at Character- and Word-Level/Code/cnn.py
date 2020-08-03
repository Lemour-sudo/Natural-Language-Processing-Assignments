#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2019-20: Homework 5
"""

import torch
import torch.nn as nn

class CNN(nn.Module):
    
    # Remember to delete the above 'pass' after your implementation
    ### YOUR CODE HERE for part 1g

    def __init__(self, e_char: int, num_filters: int, kernel_size=5, padding=1):
        """
        Instantiates ...

        @param e_char (int): number of input channels for Conv1d (is character embedding size in document).
        @param num_filters (int): number of filters (e_word = num_filters in document) (also called number of output features or number of output channels).
        @param kernel_size (int): kernel size (also called window size).
        @param padding (int): padding size.
        """
        super(CNN, self).__init__()

        self.e_char = e_char
        self.e_word = num_filters # e_word is the word_embed_size
        self.k = kernel_size
        self.pad = padding

        self.convlayer = nn.Conv1d(self.e_char, self.e_word, self.k, padding=self.pad)

    def forward(self, x_reshaped: torch.Tensor) -> torch.Tensor:
        """
        Takes a batch of raw word embeddings and returns single embedding vectors for each word. 

        @param x_reshaped (Tensor, size = batch_size * e_char * m_word): batch of raw word embeddings.

        @returns x_conv_out (Tensor, size = batch_size * word_embed_size): new transformed word embeddings.
        """
        batch_size = x_reshaped.shape[0]
        m_word = x_reshaped.shape[2]

        x_conv = self.convlayer(x_reshaped)
        # print("\n\nx_conv {} =\n{}".format(tuple(x_conv.size()), x_conv))
        assert x_conv.shape == (batch_size, self.e_word, (m_word - self.k + 1 + 2*self.pad)), \
            "x_conv is incorrect size; expected {} but got {}". \
                format((batch_size, self.e_word, (m_word - self.k + 1 + 2*self.pad)), tuple(x_conv.size()))

        maxpool = nn.MaxPool1d(kernel_size=x_conv.shape[2])
        x_conv_out = maxpool(x_conv.clamp(min=0))
        x_conv_out = torch.squeeze(x_conv_out, dim=2)
        # print("\nx_conv_out {} =\n{}".format(tuple(x_conv_out.size()), x_conv_out))
        assert x_conv_out.shape == (batch_size, self.e_word), \
            "x_conv_out is incorrect size; expected ({}, {}) but got {}". \
                format(batch_size, self.e_word, tuple(x_conv_out.size()))

        return x_conv_out


    ### END YOUR CODE

