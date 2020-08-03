#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2019-20: Homework 5
model_embeddings.py: Embeddings for the NMT model
Pencheng Yin <pcyin@cs.cmu.edu>
Sahil Chopra <schopra8@stanford.edu>
Anand Dhoot <anandd@stanford.edu>
Michael Hahn <mhahn2@stanford.edu>
"""

import torch.nn as nn
import torch

# Do not change these imports; your module names should be
#   `CNN` in the file `cnn.py`
#   `Highway` in the file `highway.py`
# Uncomment the following two imports once you're ready to run part 1(j)

from cnn import CNN
from highway import Highway


# End "do not change"

class ModelEmbeddings(nn.Module):
    """
    Class that converts input words to their CNN-based embeddings.
    """

    def __init__(self, word_embed_size, vocab):
        """
        Init the Embedding layer for one language
        @param word_embed_size (int): Embedding size (dimensionality) for the output word
        @param vocab (VocabEntry): VocabEntry object. See vocab.py for documentation.

        Hints: - You may find len(self.vocab.char2id) useful when create the embedding
        """
        super(ModelEmbeddings, self).__init__()

        ### YOUR CODE HERE for part 1h

        self.word_embed_size = word_embed_size
        self.vocab = vocab
        self.dropout = nn.Dropout(p=0.3)
        self.e_char = 50

        self.char_embeds = nn.Embedding(len(self.vocab.char2id), self.e_char)
        
        self.cnn = CNN(self.e_char, self.word_embed_size)

        self.highway = Highway(self.word_embed_size)

        ### END YOUR CODE

    def forward(self, input):
        """
        Looks up character-based CNN embeddings for the words in a batch of sentences.
        @param input: Tensor of integers of shape (sentence_length, batch_size, max_word_length) where
            each integer is an index into the character vocabulary

        @param output: Tensor of shape (sentence_length, batch_size, word_embed_size), containing the
            CNN-based embeddings for each word of the sentences in the batch
        """
        ### YOUR CODE HERE for part 1h

        x_padded = input    # not a big fan of using the keyword 'input' as variable name
        sent_len, batch_size, m_word = x_padded.size()

        x_reshaped = self.char_embeds(x_padded)      # lookup the character embeddings
        x_reshaped = x_reshaped.permute(0, 1, 3, 2)
        assert x_reshaped.size() == (sent_len, batch_size, self.e_char, m_word), \
            "x_rehaped is incorrect size; expected {} but got {}". \
                format((sent_len, batch_size, self.e_char, m_word), tuple(x_reshaped.shape))

        x_conv_out = self.cnn.forward(torch.flatten(x_reshaped, 0, 1))
        assert x_conv_out.size() == (sent_len*batch_size, self.word_embed_size), \
            "x_conv_out is incorrect size; expected {} but got {}". \
                format((sent_len*batch_size, self.word_embed_size), tuple(x_conv_out.shape))

        x_highway = self.highway.forward(x_conv_out)
        assert x_highway.size() == (sent_len*batch_size, self.word_embed_size), \
            "x_highway is incorrect size; expected {} but got {}". \
                format((sent_len*batch_size, self.word_embed_size), tuple(x_highway.shape))

        x_word_emb = self.dropout(x_highway)
        x_word_emb = torch.stack(torch.split(x_word_emb, batch_size, dim=0))
        assert x_word_emb.size() == (sent_len, batch_size, self.word_embed_size), \
            "x_word_emb is incorrect size; expected {} but got {}". \
                format((sent_len, batch_size, self.word_embed_size), tuple(x_word_emb.size()))
        # print("\nx_word_emb =\n", x_word_emb)

        return x_word_emb

        ### END YOUR CODE

