# This is a sample Python script.
import copy

import torch
from torch import nn

EPOCHS = 10
BATCH_SIZE = 32
pos = True
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

from Part1.utils import create_dev_train, plot_values


class MLP_Tagger(nn.Module):
    def __init__(self, input_size, hidden_layer_size, output_size, embed_size=50):
        super(MLP_Tagger, self).__init__()
        self.embedding = nn.Embedding(input_size, embed_size, padding_idx=0)
        self.fc1 = nn.Linear(embed_size * 5, hidden_layer_size)
        self.fc2 = nn.Linear(hidden_layer_size, output_size)

    def forward(self, x):
        x = self.embedding(x)
        x = x.view(-1, 250)
        x = self.fc1(x)
        x = nn.functional.tanh(x)
        x = nn.functional.dropout(x, training=self.training, p=0.2)
        x = self.fc2(x)
        x = nn.functional.softmax(x, dim=1)
        return x


def train(model, train_loader, optimizer, epoch):
    running_loss = 0.0
    last_loss = 0.0
    for batch_idx, data in enumerate(train_loader):

        optimizer.zero_grad()
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        loss = nn.CrossEntropyLoss()(outputs, labels)
        loss.backward()

        optimizer.step()
        running_loss += loss.item()

        if batch_idx % 1000 == 999:
            last_loss = running_loss / 1000  # loss per batch
            print('  batch {} loss: {}'.format(batch_idx + 1, last_loss))
            running_loss = 0.
    return last_loss


if __name__ == '__main__':

    if pos:
        prefix = '../pos/'
    else:
        prefix = '../ner/'

    vocab, train_data, dev, labels, test_data, test_words = create_dev_train(f'{prefix}train', f'{prefix}dev',
                                                                             f'{prefix}test')
    model = MLP_Tagger(len(vocab), 128, len(labels), embed_size=50).to(device)
    train_set = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    validation_loader = torch.utils.data.DataLoader(dev, batch_size=100, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters())
    best_loss = 1e6
    i_to_labels ={labels[l]:l for l in labels.keys()}
    dev_losses = []
    dev_accuracies = []

    best_acc = 0
    best_model = None

    for epoch in range(EPOCHS):
        print('Epoch {}/{}'.format(epoch + 1, EPOCHS))
        model.train()
        avg_loss = train(model, train_set, optimizer, epoch)
        model.eval()

        # Disable gradient computation and reduce memory consumption.
        running_vloss = 0.0
        with torch.no_grad():
            accurate = 0
            to_reduce = 0  #should be 0 for POS
            for i, vdata in enumerate(validation_loader):
                vinputs, vlabels = vdata
                vinputs = vinputs.to(device)
                vlabels = vlabels.to(device)
                voutputs = model(vinputs)
                vloss = nn.CrossEntropyLoss()(voutputs, vlabels)
                running_vloss += vloss.item()
                for k in range(len(vlabels)):
                    pred = torch.argmax(voutputs[k])

                    if pred == vlabels[k]:
                        if not pos and pred == labels['O']:
                            to_reduce += 1
                        accurate += 1
        accuracy = 100 * ((accurate - to_reduce) / (len(validation_loader.dataset) - to_reduce))
        print(
            f'validation accuracy: {accuracy}')

        avg_vloss = running_vloss / (i + 1)
        dev_losses.append(avg_vloss)
        dev_accuracies.append(accuracy)
        print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))
        if accuracy > best_acc:
            best_acc = accuracy
            best_model = copy.deepcopy(model)

    settings = 'pos' if pos else 'ner'
    plot_values(dev_losses, f'Dev Losses {settings}')
    plot_values(dev_accuracies, f'Dev Accuracies {settings}')

    #accurate = 0
    #for batch_idx, data in enumerate(train_set):
    #    inputs, labels = data
    #    inputs = inputs.to(device)
    #    labels = labels.to(device)
    #    outputs = model(inputs)
    #    for i in range(len(labels)):
    #        pred = torch.argmax(outputs[i])
    #        if pred == labels[i]:
    #            accurate += 1
    #print(f'final train accuracy: {100 * accurate / len(train_set.dataset)}')

    test_words = test_words[2:-2]
    f = open(f'test1.{settings}', 'w')

    for x, w in zip(test_data, test_words):
        x = x.to(device)
        pred = torch.argmax(best_model(x))
        label = i_to_labels[pred.item()]
        f.write(f'{w} {label}\n')

    f.close()
