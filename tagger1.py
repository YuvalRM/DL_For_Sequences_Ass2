# This is a sample Python script.
import torch
from torch import nn
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from utils import LABELS_pos,TRAIN_pos,DEV_pos,vocab_pos
hyper_parameters = {"lr": 5e-3, 'epochs': 100, 'momentum': 0.9}

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def create_tensor_with_prob_zero( tensor_size,rand_value):
    ones_tensor = torch.ones(tensor_size)
    zero_mask = torch.rand(tensor_size[0]) < rand_value
    ones_tensor[:, 2] = (1 - zero_mask.type(torch.float))
    ones_tensor=ones_tensor.int()
    return ones_tensor


class MLP_Tagger(nn.Module):
    def __init__(self, input_size, output_size, embed_size=50):
        super(MLP_Tagger, self).__init__()
        self.embedding = nn.Embedding(input_size, embed_size, padding_idx=0)
        self.fc1 = nn.Linear(embed_size*5, output_size)



    def forward(self, x):

        x = self.embedding(x)

        shape=x.shape
        x=x.view(shape[0],shape[1]*shape[2])
        x = self.fc1(x)
        x = nn.functional.tanh(x)
        x = nn.functional.softmax(x,dim=1)
        return x


def train(model, train_loader, optimizer, epoch, tb_writer):
    rand_value = 0.2  # hyper parameter for vanishing randomly words s.t we will be able to handle words not int the vocab
    running_loss = 0.0
    last_loss = 0.0
    accurate=0
    for batch_idx, data in enumerate(train_loader):

        optimizer.zero_grad()
        inputs, labels = data
        y = create_tensor_with_prob_zero(inputs.shape, rand_value)
        inputs=inputs*y
        inputs = inputs.to(device)
        labels=labels.to(device)
        outputs = model(inputs)

        loss = nn.CrossEntropyLoss()(outputs, labels)
        loss.backward()

        optimizer.step()
        running_loss += loss.item()
        if epoch % 5 ==0:
            for i in range(len(labels)):
                pred = torch.argmax(outputs[i])
                if pred == torch.argmax(labels[i]):
                    accurate += 1

        if batch_idx % 1000 == 999:
            last_loss = running_loss / 1000  # loss per batch
            print('  batch {} loss: {}'.format(batch_idx + 1, last_loss))
            tb_x = epoch * len(train_loader) + batch_idx + 1
            tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            running_loss = 0.
    if epoch % 5 == 0:
        print(f'train accuracy: {accurate / len(train_loader)}')
    return last_loss


if __name__ == '__main__':

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    writer = SummaryWriter('runs/fashion_trainer_{}'.format(timestamp))
    model = MLP_Tagger(len(vocab_pos), len(LABELS_pos), embed_size=50).to(device)
    train_set=torch.utils.data.DataLoader(TRAIN_pos, batch_size=150, shuffle=True)
    validation_loader = torch.utils.data.DataLoader(DEV_pos, batch_size=100, shuffle=True)
    epochs=hyper_parameters['epochs']
    optimizer = torch.optim.SGD(model.parameters(), lr=hyper_parameters['lr'], momentum=hyper_parameters['momentum'])
    best_loss= 1e6
    for epoch in range(epochs):
        print('Epoch {}/{}'.format(epoch + 1, epochs))
        model.train()
        avg_loss = train(model, train_set, optimizer, epoch, writer)
        model.eval()

        # Disable gradient computation and reduce memory consumption.
        running_vloss = 0.0
        with torch.no_grad():
            accurate = 0
            for i, vdata in enumerate(validation_loader):
                vinputs, vlabels = vdata
                vinputs=vinputs.to(device)
                vlabels=vlabels.to(device)
                voutputs = model(vinputs)
                vloss = nn.CrossEntropyLoss()(voutputs, vlabels)
                running_vloss += vloss
                for i in range(len(vlabels)):
                    pred=torch.argmax(voutputs[i])
                    if pred ==torch.argmax(vlabels[i]):
                        accurate+=1
            print(f'validation accuracy: {accurate/len(validation_loader)}')

        avg_vloss = running_vloss / (i + 1)
        print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))
        writer.add_scalars('Training vs. Validation Loss',
                           {'Training': avg_loss, 'Validation': avg_vloss},
                           epoch + 1)
        writer.flush()
    for  batch_idx, data in enumerate(train_set):
        inputs, labels = data
        inputs = inputs.to(device)
        labels=labels.to(device)
        outputs = model(inputs)
        for i in range(len(labels)):
            pred = torch.argmax(outputs[i])
            if pred == torch.argmax(labels[i]):
                accurate += 1
    print(f'final train accuracy: {accurate / len(train_set)}')