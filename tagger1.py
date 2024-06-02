# This is a sample Python script.
import torch
from torch import nn


def load_data(file_name):
    f = open(file_name, 'r')
    lines = f.readlines()
    data = []
    for line in lines:
        line = line.strip()
        x_y = line.split()
        if len(x_y) == 2 and len(x_y[1]) == 2:
            data.append((x_y[0], x_y[1]))
    return data


TRAIN = [(w, pos) for w, pos in load_data('./pos/train')]
vocab = {l: i for i, l in enumerate(list(sorted(set([l for l, t in TRAIN]))))}


class MLP_Tagger(nn.Module):
    def __init__(self, input_size, output_size, embed_size=50):
        super(MLP_Tagger, self).__init__()
        self.embedding = nn.Embedding(input_size+1, embed_size,padding_idx=0)
        self.fc1 = nn.Linear(embed_size, output_size)

    def forward(self, x):
        x = self.embedding(x)
        x = self.fc1(x)
        x = nn.functional.tanh(x)
        x = nn.functional.softmax(x)
        return x


if __name__ == '__main__':
    data = load_data('./pos/train')
    print('hi')
