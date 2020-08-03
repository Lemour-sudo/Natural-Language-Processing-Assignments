
"""
By Tshepo Mosoeunyane
Tests for Highway and CNN

Usage:
    run.py highway
    run.py cnn
    run.py charcoder
"""

import torch
import torch.nn as nn
import numpy as np
import sys
from docopt import docopt
import json
import time

sys.path.append("..")
from highway import Highway
from cnn import CNN
from char_decoder import CharDecoder

torch.manual_seed(0)
torch.cuda.manual_seed(0)

# device = torch.device('cuda:0')
device = torch.device('cpu')


class DummyVocab():
    def __init__(self):
        self.char2id = json.load(open('char_vocab_sanity_check.json', 'r'))
        self.id2char = {id: char for char, id in self.char2id.items()}
        self.char_pad = self.char2id['<pad>']
        self.char_unk = self.char2id['<unk>']
        self.start_of_word = self.char2id["{"]
        self.end_of_word = self.char2id["}"]



def test_highway():
    print()
    print("==="*30)
    print("\nHighway Class test")

    e_word = 3

    x_conv_out = torch.tensor(
        [
            [
                [0, 1, 1],      # sentence a's word 1
                [-1, 1, 0]      # sentence b's word 1
            ],
            [
                [1, 0, 0],      # sentence a's word 2
                [0, 1, 0]       # sentence a's word 2
            ]
        ],
        dtype=torch.float,
        device=device
    )

    sent_len = x_conv_out.shape[0]
    batch_size = x_conv_out.shape[0]

    correct_x_highway = np.array(
        [
            [
                [ 0., 0.38797045, 0.57840323],          # sentence a's word 1
                [-0.03674287, 0.4926422, 0.22739217]    # sentence b's word 1
            ],
            [
                [ 0.58957815, 0., 0.],                  # sentence a's word 2
                [ 0.24245806, 0.47267026, 0.18764845]   # sentence b's word 2
            ]
        ]
    )

    model = Highway(e_word).to(device)
    obtained_x_highway = model.forward(torch.flatten(x_conv_out, 0, 1))
    obtained_x_highway = torch.stack(torch.split(obtained_x_highway, batch_size, dim=0))
    obtained_x_highway = obtained_x_highway.cpu().detach().numpy()

    assert np.allclose(correct_x_highway, obtained_x_highway), \
        "\n\nIncorrect x_highway\n\nCorrect x_highway:\n{}\n\nYour x_highway:\n{}". \
            format(correct_x_highway, obtained_x_highway)

    print("\nx_highway =\n", obtained_x_highway)

    # # Check the weights
    # print("\nWproj weights:\n", model.Wproj.weight.cpu().detach().numpy())
    # print("\nWproj bias:\n", model.Wproj.bias.cpu().detach().numpy())
    # print("\n\nWgate weights:\n", model.Wgate.weight.cpu().detach().numpy())
    # print("\nWgate bias:\n", model.Wgate.bias.cpu().detach().numpy())

    print("\n\nHighway Test Passed!\n")
    print("==="*30)


def test_cnn():
    print()
    print("==="*30)
    print("\nCNN Class test")
    
    x_reshaped = torch.tensor(
        [
            [
                [[0, 1, 1], [-1, 1, 0]],    # sentence a's word 1
                [[0, 2, 1], [1, 0, 0]]      # sentence b's word 1      
            ],
            [
                [[0, 1, 1], [-5, 1, 0]],     # sentence a's word 2
                [[0, 2, 0], [-6, 0, 1]]     # sentence b's word 2
            ]
        ],
        dtype=torch.float,
        device=device
    )

    sent_len = x_reshaped.shape[0]
    batch_size = x_reshaped.shape[0]

    correct_x_conv_out = np.array(
        [
            [
                [0.3235975,  0.6727363 ],   # sentence a's word 1
                [0.21636295, 0.528751  ]    # sentence b's word 1      
            ],
            [
                [1.91349313, 0.71236265],   # sentence a's word 2
                [2.961207, 0.45990318]      # sentence b's word 2
            ]
        ]
    ) 

    model = CNN(2, 2, kernel_size=2).to(device)
    obtained_x_conv_out = model.forward(torch.flatten(x_reshaped, 0, 1))
    obtained_x_conv_out = torch.stack(torch.split(obtained_x_conv_out, batch_size, dim=0))
    obtained_x_conv_out = obtained_x_conv_out.cpu().detach().numpy()

    assert np.allclose(correct_x_conv_out, obtained_x_conv_out), \
        "\n\nIncorrect x_conv_out\n\nCorrect x_conv_out:\n{}\n\nYour x_conv_out:\n{}". \
            format(correct_x_conv_out, obtained_x_conv_out)

    print("\n\nx_conv_out =\n", obtained_x_conv_out, "\n")

    # # Check the weights
    # print("\nCNN weights:\n", model.convlayer.weight.cpu().detach().numpy())
    # print("\n\nCNN bias:\n", model.convlayer.bias.cpu().detach().numpy())
    
    print("\nCNN Test Passed!\n")
    print("==="*30)

def test_char_encoder():
    print("==="*30)
    print("\nCharDecoder/decode_greedy test")
    char_vocab = DummyVocab()
    HIDDEN_SIZE = 6
    EMBED_SIZE = 3
    BATCH_SIZE = 4

    h0 = torch.randn(1, BATCH_SIZE, HIDDEN_SIZE, dtype=torch.float)
    c0 = torch.randn(1, BATCH_SIZE, HIDDEN_SIZE, dtype=torch.float)

    decoder = CharDecoder(
        hidden_size=HIDDEN_SIZE,
        char_embedding_size=EMBED_SIZE,
        target_vocab=char_vocab)

    start_time = time.time()
    decodedWords = decoder.decode_greedy((h0, c0), device)
    print("\n--- decode_greedy takes %s seconds ---" % (time.time() - start_time))

    print("\n\ndecodedWords:\n{}\n".format(decodedWords))

    print("\nCharDecoder/decode_greedy Test Passed!\n")
    print("==="*30)


args = docopt(__doc__)

if args['highway']:
    test_highway()
elif args['cnn']:
    test_cnn()
elif args['charcoder']:
    test_char_encoder()
else:
    raise RuntimeError('invalid run mode')



"""
These are the linear layers weights for:
torch.manual_seed(0)
torch.cuda.manual_seed(0)

(first, cuda) torch.nn.linear(3, 3, bias=True).to(device):
    weights:
       [[-0.0043,  0.3097, -0.4752],
        [-0.4249, -0.2224,  0.1548],
        [-0.0114,  0.4578, -0.0512]]
    bias:
        [ 0.1528, -0.1745, -0.1135]

(second, cuda) torch.nn.Linear(3, 3, bias=True).to(device):
    weights:
       [[-0.5516, -0.3824, -0.2380],
        [ 0.0214,  0.2282,  0.3464],
        [-0.3914, -0.2514,  0.2097]]
    bias:
        [ 0.4794, -0.1188,  0.4320]
"""


"""
These are the Conv1d weights for:
torch.manual_seed(0)
torch.cuda.manual_seed(0)

Conv1d parameters (cuda)

torch.nn.Conv1d(2, 2, 2).cuda()

weight:
[
    [[-0.00374341,  0.2682218 ]
     [-0.41152257, -0.3679695 ]],

    [[-0.19257718,  0.13407868],
     [-0.00990659,  0.39644474]]
]

bias:
    [-0.04437202,  0.13230628]
"""