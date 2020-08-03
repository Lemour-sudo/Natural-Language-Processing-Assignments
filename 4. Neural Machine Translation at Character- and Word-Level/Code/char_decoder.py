#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2019-20: Homework 5
"""

import torch
import torch.nn as nn


class CharDecoder(nn.Module):
    def __init__(self, hidden_size, char_embedding_size=50, target_vocab=None):
        """ Init Character Decoder.

        @param hidden_size (int): Hidden size of the decoder LSTM
        @param char_embedding_size (int): dimensionality of character embeddings
        @param target_vocab (VocabEntry): vocabulary for the target language. See vocab.py for documentation.
        """
        super(CharDecoder, self).__init__()
        self.target_vocab = target_vocab
        self.charDecoder = nn.LSTM(char_embedding_size, hidden_size)
        self.char_output_projection = nn.Linear(hidden_size, len(self.target_vocab.char2id))
        self.decoderCharEmb = nn.Embedding(len(self.target_vocab.char2id), char_embedding_size,
                                           padding_idx=self.target_vocab.char_pad)

    def forward(self, input, dec_hidden=None):
        """ Forward pass of character decoder.

        @param input (Tensor): tensor of integers, shape (length, batch_size)
        @param dec_hidden (tuple(Tensor, Tensor)): internal state of the LSTM before reading the input characters. A tuple of two tensors of shape (1, batch, hidden_size)

        @returns scores (Tensor): called s_t in the PDF, shape (length, batch_size, self.vocab_size)
        @returns dec_hidden (tuple(Tensor, Tensor)): internal state of the LSTM after reading the input characters. A tuple of two tensors of shape (1, batch, hidden_size)
        """
        ### YOUR CODE HERE for part 2a
        ### TODO - Implement the forward pass of the character decoder.

        x_t = input    # not a big fan of using the keyword 'input' as variable name
        x_t = self.decoderCharEmb(x_t)

        h_t, dec_state = self.charDecoder(x_t, dec_hidden)

        s_t = self.char_output_projection(h_t)

        return s_t, dec_state

        ### END YOUR CODE

    def train_forward(self, char_sequence, dec_hidden=None):
        """ Forward computation during training.

        @param char_sequence (Tensor): tensor of integers, shape (length, batch_size). Note that "length" here and in forward() need not be the same.
        @param dec_hidden (tuple(Tensor, Tensor)): initial internal state of the LSTM, obtained from the output of the word-level decoder. A tuple of two tensors of shape (1, batch_size, hidden_size)

        @returns The cross-entropy loss (Tensor), computed as the *sum* of cross-entropy losses of all the words in the batch.
        """
        ### YOUR CODE HERE for part 2b
        ### TODO - Implement training forward pass.
        ###
        ### Hint: - Make sure padding characters do not contribute to the cross-entropy loss. Check vocab.py to find the padding token's index.
        ###       - char_sequence corresponds to the sequence x_1 ... x_{n+1} (e.g., <START>,m,u,s,i,c,<END>). Read the handout about how to construct input and target sequence of CharDecoderLSTM.
        ###       - Carefully read the documentation for nn.CrossEntropyLoss and our handout to see what this criterion have already included:
        ###             https://pytorch.org/docs/stable/nn.html#crossentropyloss

        loss = nn.CrossEntropyLoss(ignore_index=self.target_vocab.char_pad)

        targets = char_sequence[:][1:]
        char_sequence = char_sequence[:][:-1]

        s_t, _ = self.forward(char_sequence, dec_hidden)

        preds = s_t.permute(1, 0, 2).contiguous()
        targets = targets.permute(1, 0).contiguous()

        total_loss = sum([loss(pred, target) for pred, target in zip(preds, targets)])
        
        return total_loss

        ### END YOUR CODE

    def decode_greedy(self, initialStates, device, max_length=21):
        """ Greedy decoding
        @param initialStates (tuple(Tensor, Tensor)): initial internal state of the LSTM, a tuple of two tensors of size (1, batch_size, hidden_size)
        @param device: torch.device (indicates whether the model is on CPU or GPU)
        @param max_length (int): maximum length of words to decode

        @returns decodedWords (List[str]): a list (of length batch_size) of strings, each of which has length <= max_length.
                              The decoded strings should NOT contain the start-of-word and end-of-word characters.
        """

        ### YOUR CODE HERE for part 2c
        ### TODO - Implement greedy decoding.
        ### Hints:
        ###      - Use initialStates to get batch_size.
        ###      - Use target_vocab.char2id and target_vocab.id2char to convert between integers and characters
        ###      - Use torch.tensor(..., device=device) to turn a list of character indices into a tensor.
        ###      - You may find torch.argmax or torch.argmax useful
        ###      - We use curly brackets as start-of-word and end-of-word characters. That is, use the character '{' for <START> and '}' for <END>.
        ###        Their indices are self.target_vocab.start_of_word and self.target_vocab.end_of_word, respectively.

        batch_size = initialStates[0].shape[1]

        start_token = self.target_vocab.start_of_word
        end_token = self.target_vocab.end_of_word

        h_t, c_t = initialStates

        final_word_tokens = [[],]  * batch_size
        current_chars = [start_token,] * batch_size
        current_chars = torch.tensor(current_chars, dtype=torch.long, device=device)

        for t in range(max_length):
            current_chars = current_chars.unsqueeze(dim=0)
            
            scores, (h_t, c_t) = self.forward(current_chars, (h_t, c_t))
            scores = scores.squeeze(dim=0)

            soft_max = nn.Softmax(dim=1)
            probs = soft_max(scores)
            current_chars = probs.argmax(dim=1)

            final_word_tokens = [
                word + [char.tolist()] for word, char in zip(final_word_tokens, current_chars)
            ]

        # Slice the words to remove end-tokens and parts after end-tokens
        for i, word in enumerate(final_word_tokens):
            try:
                final_word_tokens[i] = word[:word.index(end_token)]
            except:
                pass

        # Convert word tokens to characters
        final_word_chars = [
            [[self.target_vocab.id2char[j] for j in word] for word in final_word_tokens]
        ]
        final_word_chars = final_word_chars[0]

        # Conver lists of strings to strings
        decodedWords = [''.join(word) for word in final_word_chars]

        return decodedWords       

        ### END YOUR CODE

