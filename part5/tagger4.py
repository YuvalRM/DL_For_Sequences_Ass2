import copy
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from part5.utils import get_train_dev, plot_values, process_file
from torch.nn.utils.rnn import pad_sequence

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Tagger4(nn.Module):
    def __init__(self,
                 num_words_embeddings,
                 num_chars_embeddings,
                 hidden_layer_size,
                 output_size,
                 word_id_to_chars_ids,
                 embed_size=50,
                 pre_trained_embeddings=False):

        super(Tagger4, self).__init__()
        self.word_id_to_chars_ids = word_id_to_chars_ids
        self.word_embedding = nn.Embedding(num_words_embeddings, embed_size, padding_idx=0)
        self.chars_embedding = nn.Embedding(num_chars_embeddings, embed_size, padding_idx=0)
        self.cnn1 = nn.Conv1d(embed_size, embed_size, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(embed_size * 5, hidden_layer_size)
        self.fc2 = nn.Linear(hidden_layer_size, output_size)
        if pre_trained_embeddings:
            self.word_embedding.weight.data.copy_(torch.from_numpy(np.loadtxt("../wordVectors.txt")))

    def embed_chars(self, tensor):
        keys = tensor.tolist()  # Convert tensor to list
        chars_emebeddings = pad_sequence([self.chars_embedding(self.word_id_to_chars_ids[key]) for key in keys], batch_first=True)
        return chars_emebeddings.transpose(1, 2)

    def forward(self, x):
        chars0 = self.cnn1(self.embed_chars(x[:, 0]))
        chars1 = self.cnn1(self.embed_chars(x[:, 1]))
        chars2 = self.cnn1(self.embed_chars(x[:, 2]))
        chars3 = self.cnn1(self.embed_chars(x[:, 3]))
        chars4 = self.cnn1(self.embed_chars(x[:, 4]))
        chars = torch.cat((chars0, chars1, chars2, chars3, chars4), dim=2)

        stride = kernel_size = chars.shape[2] // 5
        max_pool = nn.MaxPool1d(stride=stride, kernel_size=kernel_size)
        chars = max_pool(chars)

        words = self.word_embedding(x).transpose(1, 2)
        x = words + chars
        x = x.transpose(1,2).view(-1, 250)
        x = self.fc1(x)
        x = nn.functional.tanh(x)
        x = self.fc2(x)
        x = nn.functional.softmax(x, dim=1)
        return x




# Hyperparameters
hidden_size = 128
learning_rate = 0.001
batch_size = 16
num_epochs = 1

def train(pos=True, pre_trained_embeddings=True):
    mode = 'pos' if pos else 'ner'
    pre_trained = 'pre_trained' if pre_trained_embeddings else 'random'

    train_file_path = f'../{mode}/train'
    dev_file_path = f'../{mode}/dev'
    test_file_path = f'../{mode}/test'
    fig_acc = f'fig_acc_{mode}_{pre_trained}.png'
    fig_loss = f'fig_loss_{mode}_{pre_trained}.png'


    train_dataset, dev_dataset, test_dataset, num_words_embeddings, num_chars_embeddings, label_to_id, label_id_to_label, test_words, word_id_to_chars_ids =\
        get_train_dev(train_file_path, dev_file_path, test_file_path)
    num_labels = len(label_to_id) - 1

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False)

    # Initialize the model, loss function, and optimizer
    model = Tagger4(num_words_embeddings, num_chars_embeddings, hidden_size, num_labels, word_id_to_chars_ids, pre_trained_embeddings=pre_trained_embeddings)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    dev_losses, dev_acc = [], []
    best_dev_acc = -1
    best_model = None
    if not pos:
        O_id = label_to_id['O']

    # Training loop
    for epoch in range(num_epochs):
        batch_idx, running_loss = 0, 0.
        for batch in train_loader:
            samples, labels = batch
            samples, labels = samples.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(samples)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if batch_idx % 1000 == 999:
                last_loss = running_loss / 1000  # loss per batch
                print('  batch {} loss: {}'.format(batch_idx + 1, last_loss))
                running_loss = 0.

            batch_idx += 1

        model.eval()
        correct = 0
        total = 0
        loss_dev = 0.
        with torch.no_grad():
            for batch in dev_loader:
                samples, labels = batch
                samples, labels = samples.to(device), labels.to(device)
                outputs = model(samples)
                _, predicted = torch.max(outputs.data, 1)
                loss_dev += criterion(outputs, labels).item()

                # for evaluating NER, we ignore the 'O' label if there is the predicted is also 'O'
                if not pos:
                    mask = (predicted != O_id) | (labels != O_id)
                    predicted = predicted[mask]
                    labels = labels[mask]
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        loss_dev = loss_dev / len(dev_loader)
        accuracy = 100 * correct / total
        dev_losses.append(loss_dev)
        dev_acc.append(accuracy)

        if accuracy > best_dev_acc:     # early stopping
            best_dev_acc = accuracy
            best_model = copy.deepcopy(model)

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss_dev:.4f}, Accuracy: {accuracy:.2f}%')

    plot_values(fig_loss, dev_losses, 'Dev Loss')
    plot_values(fig_acc, dev_acc, 'Dev Accuracy')

    return best_model, label_id_to_label, test_dataset, test_words


def test(test_dataset, model, label_id_to_label, test_words):
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    model.eval()
    predictions = []
    i = 2
    with torch.no_grad():
        for batch in test_loader:
            samples, _ = batch
            samples = samples.to(device)
            outputs = model(samples)
            _, predicted = torch.max(outputs.data, 1)
            predictions.append((test_words[i][0], label_id_to_label[predicted.item()]))
            i += 1
    return predictions


# NER
best_model, label_id_to_label, test_dataset, test_words = train(pos=False, pre_trained_embeddings=True)
predictions = test(test_dataset, best_model, label_id_to_label, test_words)
process_file('../ner/test','test4.ner', predictions)

best_model, label_id_to_label, test_dataset, test_words = train(pos=False, pre_trained_embeddings=False)
predictions = test(test_dataset, best_model, label_id_to_label, test_words)
process_file('../ner/test','test4.ner', predictions)

# POS
best_model, label_id_to_label, test_dataset, test_words = train(pos=True, pre_trained_embeddings=True)
predictions = test(test_dataset, best_model, label_id_to_label, test_words)
process_file('../pos/test','test4.pos', predictions)

best_model, label_id_to_label, test_dataset, test_words = train(pos=True, pre_trained_embeddings=False)
predictions = test(test_dataset, best_model, label_id_to_label, test_words)
process_file('../pos/test','test4.pos', predictions)

