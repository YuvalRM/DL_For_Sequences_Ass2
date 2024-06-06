import numpy as np
import torch
from matplotlib import pyplot as plt


def load_labes(file_name):
    f = open(file_name, 'r')
    lines = f.readlines()
    labes = {}
    indexes = {}
    i = 0
    for line in lines:
        line = line.strip()
        x_y = line.split()

        if len(x_y) == 2 and x_y[1] not in labes.keys():
            labes[x_y[1]] = i
            indexes[i] = x_y[1]
            i += 1
    f.close()
    return labes,indexes


def load_data(file_name, strat_token='<s>', end_token='</s>'):
    f = open(file_name, 'r')
    lines = f.readlines()
    data = []
    data.append((strat_token, 'POS'))
    data.append((strat_token,  'POS'))
    for line in lines:
        line = line.strip()
        x_y = line.split()

        if len(x_y) == 2:
            data.append((x_y[0], x_y[1]))
    f.close()
    data.append((end_token,  'POS'))
    data.append((end_token,  'POS'))
    return data


def convert_data_to_fives(data):
    new_data = []
    for i in range(2, len(data) - 2):
        num1 = data[i - 2][0]
        num2 = data[i - 1][0]
        num4 = data[i + 1][0]
        num5 = data[2 + i][0]
        num3 = data[i][0]
        new_data.append((torch.tensor([num1, num2, num3, num4, num5]), data[i][1]))
    return new_data


def number_representation(w, w_to_i):
    if all(ch.isdigit() or ch == '.' or ch == '+' or ch == '-' for ch in w):
        pattern = ""

        # Replace each character with 'DG'
        for ch in w:
            pattern += 'DG' if ch.isdigit() else ch

        pattern = pattern if pattern in w_to_i else 'NNNUMMM'
        return pattern

    elif all(ch.isdigit() or ch == ',' for ch in w) and any(ch.isdigit() for ch in w):
        return "NNNUMMM"

    return None


def convert_to_data(data_to_label, w_to_i, l_2_i, unknown_token='UUUNKKK'):
    data = []
    for w, l in data_to_label:
        if w in w_to_i.keys():
            index = w_to_i[w]
        else:
            if w.lower() in w_to_i.keys():
                index = w_to_i[w.lower()]
            else:
                if number_representation(w, w_to_i) in w_to_i.keys():
                    index = w_to_i[number_representation(w, w_to_i)]
                else:
                    index = w_to_i[unknown_token]
        label_id = l_2_i[l]
        data.append((index, label_id))

    return data


def create_dev_train(train_file, dev_file, w_to_i):
    l_2_i, i_2_l = load_labes(train_file)

    train_2_label = load_data(train_file)
    dev_2_label = load_data(dev_file)

    train_data = convert_to_data(train_2_label, w_to_i, l_2_i)
    dev_data = convert_to_data(dev_2_label, w_to_i, l_2_i)
    train_data = convert_data_to_fives(train_data)
    dev_data = convert_data_to_fives(dev_data)

    return train_data, dev_data, l_2_i, i_2_l



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

    # Show the plot
    plt.show()
    plt.savefig(fname=y_label)