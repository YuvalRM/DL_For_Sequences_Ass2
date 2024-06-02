# This is a sample Python script.
import torch
from torch import nn
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from utils import LABELS_pos,TRAIN_pos,DEV_pos,vocab_pos
hyper_parameters = {"lr": 1e-2, 'epochs': 30, 'momentum': 0.9}






class MLP_Tagger(nn.Module):
    def __init__(self, input_size, output_size, embed_size=50):
        super(MLP_Tagger, self).__init__()
        self.embedding = nn.Embedding(input_size, embed_size, padding_idx=0)
        self.fc1 = nn.Linear(embed_size*5, output_size)
        self.rand_value = 0.08  #hyper parameter for vanishing randomly words s.t we will be able to handle words not int the vocab

    def forward(self, x):
        x = self.embedding(x)
        x=x.view(-1,250)
        x = self.fc1(x)
        x = nn.functional.tanh(x)
        x = nn.functional.softmax(x)
        return x


def train(model, device, train_loader, optimizer, epoch, tb_writer):
    running_loss = 0.0
    last_loss = 0.0
    model.to(device)
    for batch_idx, data in enumerate(train_loader):
        optimizer.zero_grad()
        inputs, labels = data
        inputs = inputs.to(device)
        labels=labels.to(device)
        outputs = model(inputs)

        loss = nn.CrossEntropyLoss()(outputs, labels)
        loss.backward()

        optimizer.step()
        running_loss += loss.item()
        if batch_idx % 1000 == 999:
            last_loss = running_loss / 1000  # loss per batch
            print('  batch {} loss: {}'.format(batch_idx + 1, last_loss))
            tb_x = epoch * len(train_loader) + batch_idx + 1
            tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            running_loss = 0.
    return last_loss


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    writer = SummaryWriter('runs/fashion_trainer_{}'.format(timestamp))
    model = MLP_Tagger(len(vocab_pos), len(LABELS_pos), embed_size=50)
    train_set=torch.utils.data.DataLoader(TRAIN_pos, batch_size=100, shuffle=True)
    validation_loader = torch.utils.data.DataLoader(DEV_pos, batch_size=100, shuffle=True)
    epochs=hyper_parameters['epochs']
    optimizer = torch.optim.SGD(model.parameters(), lr=hyper_parameters['lr'], momentum=hyper_parameters['momentum'])
    best_loss= 1e6
    for epoch in range(epochs):
        print('Epoch {}/{}'.format(epoch + 1, epochs))
        model.train()
        avg_loss = train(model, device, train_set, optimizer, epoch, writer)
        model.eval()

        # Disable gradient computation and reduce memory consumption.
        running_vloss = 0.0
        with torch.no_grad():
            for i, vdata in enumerate(validation_loader):
                vinputs, vlabels = vdata
                vinputs=vinputs.to(device)
                vlabels=vlabels.to(device)
                voutputs = model(vinputs)
                vloss = nn.CrossEntropyLoss()(voutputs, vlabels)
                running_vloss += vloss
        avg_vloss = running_vloss / (i + 1)
        print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))
        writer.add_scalars('Training vs. Validation Loss',
                           {'Training': avg_loss, 'Validation': avg_vloss},
                           epoch + 1)
        writer.flush()