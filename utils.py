import numpy as np
import torch
def relevant(symbol):
    return symbol not in ['""', '``', "''", '#', '$', '(', ')', ',', ':', '.', ';']


def load_labes(file_name):
    f = open(file_name, 'r')
    lines = f.readlines()
    labes = {}
    i = 0
    for line in lines:
        line = line.strip()
        x_y = line.split()
        if len(x_y) == 2 and relevant(x_y[1]) and x_y[1] not in labes.keys():
            labes[x_y[1]] = i
            i += 1
    return labes





def load_data(file_name):
    f = open(file_name, 'r')
    lines = f.readlines()
    data = []
    for line in lines:
        line = line.strip()
        x_y = line.split()
        if len(x_y) == 2 and relevant(x_y[1]):
            data.append((x_y[0], x_y[1]))
    return data


def create_one_hot_vector_numpy(k, n):
    vector = np.zeros(k)
    vector[n] = 1
    return vector

LABELS_pos = load_labes('./pos/train')
Word_to_vec_pos = [(w, create_one_hot_vector_numpy(len(LABELS_pos), LABELS_pos[pos])) for w, pos in load_data('./pos/train')]
vocab_pos = {l: i + 1 for i, l in enumerate(list(sorted(set([l for l, t in Word_to_vec_pos]))))}
vocab_pos[None] = 0  #for the edges and words not in the dataset
DEV_pos = [ (vocab_pos[w], create_one_hot_vector_numpy(len(LABELS_pos), LABELS_pos[pos])) if w in vocab_pos.keys() else (vocab_pos[None], create_one_hot_vector_numpy(len(LABELS_pos), LABELS_pos[pos])) for w, pos in load_data('./pos/dev')]
TRAIN_pos = [(vocab_pos[w],vec) for w,vec in Word_to_vec_pos]

def convert_data_to_fives(data):
    new_data = []
    for i in range(len(data)):
        num1,num2,num3,num4,num5= -1,-1,-1,-1,-1
        if i ==0:
            num2=0
        if i<=1:
            num1=0
        if i >= len(data)-2:
            num5=0
        if i==len(data)-1:
            num4 =0
        if num1==-1:
            num1=data[i-2][0]
        if num2 == -1:
            num2 = data[i-1][0]
        if num4==-1:
            num4=data[i+1][0]
        if num5 == -1:
            num5 = data[2 + i][0]

        num3=data[i][0]
        new_data.append((torch.tensor([num1, num2, num3, num4, num5]),data[i][1]))
    return new_data

TRAIN_pos=convert_data_to_fives(TRAIN_pos)
DEV_pos =convert_data_to_fives(DEV_pos)