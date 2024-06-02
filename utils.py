import numpy as np

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
TRAIN_pos = [(w, create_one_hot_vector_numpy(len(LABELS_pos), LABELS_pos[pos])) for w, pos in load_data('./pos/train')]
DEV_pos = [(w, create_one_hot_vector_numpy(len(LABELS_pos), LABELS_pos[pos])) for w, pos in load_data('./pos/dev')]
vocab_pos = {l: i + 1 for i, l in enumerate(list(sorted(set([l for l, t in TRAIN_pos]))))}
vocab_pos[None] = 0  #for the edges and words not in the dataset