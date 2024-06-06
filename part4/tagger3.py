import copy

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from part4.utils import get_train_dev, plot_values

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Tagger3(nn.Module):
    def __init__(self,
                 num_words_embeddings,
                 num_prefixes_embeddings,
                 num_suffixes_embeddings,
                 hidden_layer_size,
                 output_size,
                 embed_size=50):

        super(Tagger3, self).__init__()
        self.word_embedding = nn.Embedding(num_words_embeddings, embed_size, padding_idx=0)
        self.prefix_embedding = nn.Embedding(num_prefixes_embeddings, embed_size, padding_idx=0)
        self.suffix_embedding = nn.Embedding(num_suffixes_embeddings, embed_size, padding_idx=0)
        self.fc1 = nn.Linear(embed_size * 5, hidden_layer_size)
        self.fc2 = nn.Linear(hidden_layer_size, output_size)

    def forward(self, x):
        words = self.word_embedding(x[:, 0, :])
        prefixes = self.prefix_embedding(x[:, 1, :])
        suffixes = self.suffix_embedding(x[:, 2, :])
        x = words + prefixes + suffixes
        x = x.view(-1, 250)
        x = self.fc1(x)
        x = nn.functional.tanh(x)
        x = self.fc2(x)
        x = nn.functional.softmax(x, dim=1)
        return x




# Hyperparameters
hidden_size = 128
learning_rate = 0.001
batch_size = 16
num_epochs = 10

def train(pos=True):
    if pos:
        train_file_path = '../pos/train'
        dev_file_path = '../pos/dev'
    else:
        train_file_path = '../ner/train'
        dev_file_path = '../ner/dev'

    train_dataset, dev_dataset, num_words_embeddings, num_prefixes_embeddings, num_suffixes_embeddings, num_labels, label_id_to_label =\
        get_train_dev(train_file_path, dev_file_path)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False)

    # Initialize the model, loss function, and optimizer
    model = Tagger3(num_words_embeddings, num_prefixes_embeddings, num_suffixes_embeddings, hidden_size, num_labels)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    dev_losses, dev_acc = [], []
    best_dev_acc = -1
    best_model = None

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

                total += labels.size(0)
                loss_dev = criterion(outputs, labels).item()
                correct += (predicted == labels).sum().item()

        loss_dev = loss_dev / len(dev_loader)
        accuracy = 100 * correct / total
        dev_losses.append(loss_dev)
        dev_acc.append(accuracy)

        if accuracy > best_dev_acc:
            best_dev_acc = accuracy
            best_model = copy.deepcopy(model)

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss_dev:.4f}, Accuracy: {accuracy:.2f}%')

    plot_values(dev_losses, 'Dev Loss')
    plot_values(dev_acc, 'Dev Accuracy')

    return best_model, label_id_to_label

def test(pos=True):
    if pos:
        test_file_path = '../pos/test'
    else:
        test_file_path = '../ner/test'

    test_dataset, _, num_words_embeddings, num_prefixes_embeddings, num_suffixes_embeddings, num_labels, label_id_to_label =\
        get_train_dev(test_file_path, test_file_path)

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model, label_id_to_label = train(pos)

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in test_loader:
            samples, labels = batch
            samples, labels = samples.to(device), labels.to(device)
            outputs = model(samples)
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Test Accuracy: {accuracy:.2f}%')

    return label_id_to_label


best_model, label_id_to_label = train(pos=True)
