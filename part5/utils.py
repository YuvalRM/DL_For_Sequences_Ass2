import os

import torch
import matplotlib.pyplot as plt
from torch.nn.utils.rnn import pad_sequence

UNKNOWN = 'uuunkkk'
PADDING = 'ppadddd'

def load_labes(file_name):
    f = open(file_name, 'r')
    lines = f.readlines()
    labes = {}
    i = 0
    for line in lines:
        line = line.strip()
        x_y = line.split()
        if len(x_y) == 2 and x_y[1] not in labes.keys():
            labes[x_y[1]] = i
            i += 1
    f.close()
    labes[UNKNOWN] = i
    return labes

def load_data(file_name, is_label=True):
    f = open(file_name, 'r')
    lines = f.readlines()
    data = []

    # TEST
    if not is_label:
        for line in lines:
            x = line.strip()
            if len(x) >= 1:
                data.append((x.lower(), UNKNOWN))

    # DEV or TRAIN
    else:
        for line in lines:
            line = line.strip()
            x_y = line.split()
            if len(x_y) == 2:
                data.append((x_y[0].lower(), x_y[1]))
    f.close()
    return data


def create_char_to_id(vocabulary):
    # Initialize the character_to_id dictionary
    character_to_id = {UNKNOWN: 1, PADDING: 0}

    # Assign a unique ID to each character in the vocabulary
    current_id = 2
    for vocab_word in vocabulary:
        for char in vocab_word:
            if char not in character_to_id:
                character_to_id[char] = current_id
                current_id += 1

    return character_to_id


def get_unique_x_ids(tuples_list):
    unique_x = {UNKNOWN: 0}
    current_id = 1
    for x, y in tuples_list:
        if x not in unique_x:
            unique_x[x] = current_id
            current_id += 1
    return unique_x


def generate_suffix_prefix_dicts(word_to_id):
    suffix_to_id = {UNKNOWN: 0}
    prefix_to_id = {UNKNOWN: 0}

    # Initialize counters for suffix and prefix IDs
    suffix_counter = 1
    prefix_counter = 1

    for word in word_to_id:
        if len(word) >= 3:
            prefix = word[:3]
            suffix = word[-3:]
        else:
            prefix = suffix = word

        # Assign unique IDs to prefixes
        if prefix not in prefix_to_id:
            prefix_to_id[prefix] = prefix_counter
            prefix_counter += 1

        # Assign unique IDs to suffixes
        if suffix not in suffix_to_id:
            suffix_to_id[suffix] = suffix_counter
            suffix_counter += 1

    return prefix_to_id, suffix_to_id

def get_id(x, x_to_id):
    if x in x_to_id:
        return x_to_id[x]
    return x_to_id[UNKNOWN]

def get_ids(word, word_to_id, prefix_to_id, suffix_to_id):
    if len(word) >= 3:
        prefix = word[:3]
        suffix = word[-3:]
    else:
        prefix = suffix = word

    word_id = get_id(word, word_to_id)
    prefix_id = get_id(prefix, prefix_to_id)
    suffix_id = get_id(suffix, suffix_to_id)

    return word_id, prefix_id, suffix_id

def get_chars_ids(word, char_to_id, word_id, word_id_to_chars_ids):
    char_ids = []
    for char in word:
        char_ids.append(get_id(char, char_to_id))
    word_id_to_chars_ids[word_id] = torch.tensor(char_ids)
    return torch.tensor(char_ids)

def get_dataset(word_label_dataset, word_to_id, label_to_id, char_to_id, word_id_to_chars_ids):
    dataset = []

    # these for the edges
    word_label_dataset.insert(0, ("<###>", ""))
    word_label_dataset.insert(0, ("<$$$>", ""))
    word_label_dataset.append(("%%%", ""))
    word_label_dataset.append(("^^^", ""))

    for i in range(2, len(word_label_dataset) - 2):
        label = word_label_dataset[i][1]

        word1 = word_label_dataset[i - 2][0]
        word2 = word_label_dataset[i - 1][0]
        word3 = word_label_dataset[i][0]
        word4 = word_label_dataset[i + 1][0]
        word5 = word_label_dataset[i + 2][0]

        word1_id = get_id(word1, word_to_id)
        word2_id = get_id(word2, word_to_id)
        word3_id = get_id(word3, word_to_id)
        word4_id = get_id(word4, word_to_id)
        word5_id = get_id(word5, word_to_id)

        word1_char_ids = get_chars_ids(word1, char_to_id, word1_id, word_id_to_chars_ids)
        word2_char_ids = get_chars_ids(word2, char_to_id, word2_id, word_id_to_chars_ids)
        word3_char_ids = get_chars_ids(word3, char_to_id, word3_id, word_id_to_chars_ids)
        word4_char_ids = get_chars_ids(word4, char_to_id, word4_id, word_id_to_chars_ids)
        word5_char_ids = get_chars_ids(word5, char_to_id, word5_id, word_id_to_chars_ids)

        x = torch.tensor([word1_id, word2_id, word3_id, word4_id, word5_id])
        y = label_to_id[label]

        dataset.append((x, y))

    return dataset


def process_file(file_path, new_file_path, word_list):
    # Read the content of the original file
    with open(file_path, 'r') as file:
        lines = file.readlines()
    lines = [s for s in lines if s.strip() != '']

    # Make sure we have enough items in the list to process the lines
    if len(word_list) < len(lines):
        raise ValueError("The list is shorter than the number of lines in the file.")

    # Process each line and append the corresponding list item
    with open(new_file_path, 'w') as new_file:
        for line, word in zip(lines, word_list):
            line = line.strip()
            if len(line) == 0:
                continue
            new_line = f"{line.strip()} {word[1]}\n"
            new_file.write(new_line)

    print(f"Processed file saved as: {new_file_path}")
def get_train_dev(train_file_path, dev_file_path, test_file_path):
    label_to_id = load_labes(train_file_path)
    word_label_train = load_data(train_file_path)
    word_label_dev = load_data(dev_file_path)
    word_label_test = load_data(test_file_path, is_label=False)

    vocab = load_data("../vocab.txt", is_label=False)
    char_to_id = create_char_to_id(vocab)

    word_to_id = get_unique_x_ids(vocab)
    word_id_to_chars_ids = {}

    train_dataset = get_dataset(word_label_train, word_to_id, label_to_id, char_to_id, word_id_to_chars_ids)
    dev_dataset = get_dataset(word_label_dev, word_to_id, label_to_id, char_to_id, word_id_to_chars_ids)
    test_dataset = get_dataset(word_label_test, word_to_id, label_to_id, char_to_id, word_id_to_chars_ids)

    label_id_to_label = {v: k for k, v in label_to_id.items()}

    return (train_dataset,
            dev_dataset,
            test_dataset,
            len(word_to_id),
            len(char_to_id),
            label_to_id,
            label_id_to_label,
            word_label_test,
            word_id_to_chars_ids)


def plot_values(path, values, y_label):
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

    # Save the plot
    plt.savefig(path)
