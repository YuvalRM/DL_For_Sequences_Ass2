# This is a sample Python script.
import torch
from torch import nn


# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

class MLP_Tagger(nn.Module):
    def __init__(self, input_size, output_size):
        super(MLP_Tagger, self).__init__()
        self.fc1 = nn.Linear(input_size, output_size)

    def embbed(self,x):
        return x
    def forward(self, wiM2,wiM1,wi,wiP1,wiP2):
        x=torch.cat((self.embbed(wiM2),self.embbed(wiM1),self.embbed(wi),self.embbed(wiP1),self.embbed(wiP2)), dim=-1)
        x = self.fc1(x)
        x = nn.functional.tanh(x)
        x=nn.functional.softmax(x)
        return x


def load_data(file_name):
    f = open(file_name, 'r')
    lines = f.readlines()
    data = []
    for line in lines:
        line = line.strip()
        x_y = line.split()
        data.append(x_y)
    return data


if __name__ == '__main__':
    data = load_data('./pos/train')
    print('hi')
