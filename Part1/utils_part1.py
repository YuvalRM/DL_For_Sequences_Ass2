import numpy as np
import torch
from matplotlib import pyplot as plt


def load_labes(file_name):
    f = open(file_name, 'r')
    lines = f.readlines()
    labels = {}
    i = 0
    for line in lines:
        line = line.strip()
        x_y = line.split()

        if len(x_y) == 2 and x_y[1] not in labels.keys():
            labels[x_y[1]] = i
            i += 1
    f.close()
    return labels


def load_data(file_name):
    f = open(file_name, 'r')
    lines = f.readlines()
    data = []
    for line in lines:
        line = line.strip()
        x_y = line.split()

        if len(x_y) == 2:
            data.append((x_y[0], x_y[1]))
    f.close()
    return data


def convert_data_to_fives(data):
    new_data = []
    for i in range(len(data)):
        num1, num2, num3, num4, num5 = -1, -1, -1, -1, -1
        if i == 0:
            num2 = 1
        if i <= 1:
            num1 = 1
        if i >= len(data) - 2:
            num5 = 2
        if i == len(data) - 1:
            num4 = 2
        if num1 == -1:
            num1 = data[i - 2][0]
        if num2 == -1:
            num2 = data[i - 1][0]
        if num4 == -1:
            num4 = data[i + 1][0]
        if num5 == -1:
            num5 = data[2 + i][0]

        num3 = data[i][0]
        new_data.append((torch.tensor([num1, num2, num3, num4, num5]), data[i][1]))
    return new_data


def load_train(file_name, strat_token='<s>', end_token='<end>'):
    f = open(file_name, 'r')
    lines = f.readlines()
    data = []
    data.append(strat_token)
    data.append(strat_token)
    for line in lines:
        line = line.strip()
        x_y = line.split()

        if len(x_y) >= 1 and (x_y[0]) != '\n':
            data.append(x_y[0])
    f.close()
    data.append(end_token)
    data.append(end_token)
    return data


def convert_to_test(train_data, w_to_i, unknown_token='<unknown>'):
    data = []
    for w in train_data:
        if w in w_to_i.keys():
            index = w_to_i[w]
        else:
            if w.lower() in w_to_i.keys():
                index = w_to_i[w.lower()]
            else:
                index = w_to_i[unknown_token]
        data.append(index, )

    return data


def convert_test_to_fives(data):
    new_data = []
    for i in range(2, len(data) - 2):
        num1 = data[i - 2]
        num2 = data[i - 1]
        num4 = data[i + 1]
        num5 = data[2 + i]
        num3 = data[i]
        new_data.append(torch.tensor([num1, num2, num3, num4, num5]))
    return new_data


def create_dev_train(train_file, dev_file, test_file):
    LABELS = load_labes(train_file)
    Word_to_lable_id = [(w, LABELS[ner]) for w, ner in load_data(train_file)]
    i_to_labels=[(w, LABELS[ner]) for w, ner in load_data(train_file)]
    word_to_id = {l: i + 3 for i, l in enumerate(list(sorted(set([l for l, t in Word_to_lable_id]))))}
    word_to_id['<start>'] = 1  # for the edges and words not in the dataset
    word_to_id['<end>'] = 2
    word_to_id['<unknown>'] = 0
    DEV = [(word_to_id[w], LABELS[label_id]) if w in word_to_id.keys() else (word_to_id['<unknown>'], LABELS[label_id])
           for w, label_id in load_data(dev_file)]
    TRAIN = [(word_to_id[w], label) for w, label in Word_to_lable_id]
    TRAIN = convert_data_to_fives(TRAIN)
    DEV = convert_data_to_fives(DEV)
    test_words = load_train(test_file)
    test_data = convert_test_to_fives(convert_to_test(load_train(test_file), word_to_id))
    return word_to_id, TRAIN, DEV, LABELS, test_data, test_words


def plot_values(values, y_label):
    """
    Plots the given values with 'Epochs' as the x-axis label and y_label as the y-axis label.

    Args:
    values (list): A list of values to plot.
    y_label (str): The label for the y-axis.
    """
    # Generate the x values (epochs)
    epochs = list(range(1, len(values) + 1))

    # Create the plot
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, values, marker='o', linestyle='-', color='b')

    # Set the labels
    plt.xlabel('Epochs')
    plt.ylabel(y_label)

    # Set the title
    plt.title(f'{y_label} vs Epochs')

    # Show the grid
    plt.grid(True)

    plt.savefig(fname=f"./{y_label}_part_1")
    # Show the plot
    plt.show()
