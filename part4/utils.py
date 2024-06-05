import torch

UNKNOWN = '<unknown>'

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
    return labes

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


def get_dataset(word_label_dataset, word_to_id, label_to_id, prefix_to_id, suffix_to_id):
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

        word1_id, word1_prefix_id, word1_suffix_id = get_ids(word1, word_to_id, prefix_to_id, suffix_to_id)
        word2_id, word2_prefix_id, word2_suffix_id = get_ids(word2, word_to_id, prefix_to_id, suffix_to_id)
        word3_id, word3_prefix_id, word3_suffix_id = get_ids(word3, word_to_id, prefix_to_id, suffix_to_id)
        word4_id, word4_prefix_id, word4_suffix_id = get_ids(word4, word_to_id, prefix_to_id, suffix_to_id)
        word5_id, word5_prefix_id, word5_suffix_id = get_ids(word5, word_to_id, prefix_to_id, suffix_to_id)

        sample = torch.tensor([
            [word1_id, word2_id, word3_id, word4_id, word5_id],
            [word1_prefix_id, word2_prefix_id, word3_prefix_id, word4_prefix_id, word5_prefix_id],
            [word1_suffix_id, word2_suffix_id, word3_suffix_id, word4_suffix_id, word5_suffix_id],
        ])
        label = label_to_id[label]

        dataset.append((sample, label))

    return dataset

def get_train_dev(train_file_path, dev_file_path):
    label_to_id = load_labes(train_file_path)
    word_label_train = load_data(train_file_path)
    word_label_dev = load_data(dev_file_path)

    word_to_id = get_unique_x_ids(word_label_train)
    prefix_to_id, suffix_to_id = generate_suffix_prefix_dicts(word_to_id)

    train_dataset = get_dataset(word_label_train, word_to_id, label_to_id, prefix_to_id, suffix_to_id)
    dev_dataset = get_dataset(word_label_dev, word_to_id, label_to_id, prefix_to_id, suffix_to_id)

    return train_dataset, dev_dataset




