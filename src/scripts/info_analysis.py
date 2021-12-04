from argparse import ArgumentParser

import yaml
from src.utils.training_utils import choose_task_classes, choose_dataset_class, choose_model_class, choose_probe_class
from tqdm import tqdm
import torch.nn as nn
from scipy.ndimage.filters import gaussian_filter1d
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt


class Net(nn.Module):
    def __init__(self, input1_size, input2_size):
        self.H = 128
        self.dim1 = input1_size
        self.dim2 = input2_size
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input1_size, int(self.H / 2))
        self.fc2 = nn.Linear(input2_size, int(self.H / 2))
        self.fc3 = nn.Linear(self.H, self.H)
        self.fc4 = nn.Linear(self.H, 1)

    def forward(self, x, y):
        x1 = self.fc1(x)
        batchlen, seqlen, rank = x1.size()
        norms1 = torch.bmm(x1.view(batchlen * seqlen, 1, rank),
                          x1.view(batchlen * seqlen, rank, 1))
        norms1 = norms1.view(batchlen, seqlen)
        if len(y.shape) == 3:
            x2 = self.fc2(y)
            _, x2_len, _ = x2.size()
            norms2 = torch.bmm(x2.view(batchlen * seqlen, 1, rank),
                              x2.view(batchlen * seqlen, rank, 1))
            norms2 = norms2.view(batchlen, seqlen)
        else:
            padded_y = F.pad(y, (0, self.dim2 - y.size()[1]))
            x2 = self.fc2(padded_y)
            _, x2_len = x2.size()
            norms2 = x2
        h1 = F.relu(torch.cat([norms1, norms2], dim=1))
        h1 = F.pad(h1, (0, self.H - (seqlen + x2_len)))
        h2 = self.fc3(h1).clamp(min=0)
        h3 = self.fc4(h2)
        return h3


def get_info(args):
    dataset_class = choose_dataset_class(args)
    task_class, reporter_class, loss_class = choose_task_classes(args)
    task = task_class()
    expt_dataset = dataset_class(args, task)

    train_dataloader = expt_dataset.get_train_dataloader()
    dim1 = 768  # For full z
    # dim1 = int(768 / 2)
    # dim2 = 768 - dim1  # For x_second
    # dim2 = 75  # For depth vector.
    dim2 = 5  # For depth vector.

    cutoff = dim2
    model = Net(dim1, dim2)

    model.to(device)
    num_epochs = 50
    tracked_info = []
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    running_avg = 0
    for epoch in range(num_epochs):
        print("Running avg", running_avg)
        for batch in tqdm(train_dataloader, desc='[training_batch'):
            optimizer.zero_grad()
            x_sample = batch[0]
            x_first = x_sample[:, :cutoff, :dim1]
            x_second = x_sample[:, :cutoff, dim1:]

            y_sample = batch[1]
            # var1 = x_first
            # var1 = x_second
            var1 = x_sample[:, :cutoff, :]

            # var2 = x_second
            var2 = y_sample

            batchlen, seqlen, rank = var1.size()
            assert var2.shape[-1] == dim2 or len(var2.shape) == 2, "Var2 shape " + str(var2.shape)
            var2_shuffle = torch.Tensor(np.random.permutation(var2.cpu().numpy())).to(device)
            pred_xy = model(var1, var2)
            pred_x_y = model(var1, var2_shuffle)
            ret = torch.mean(pred_xy) - torch.log(torch.mean(torch.exp(pred_x_y)))
            running_avg = 0.95 * running_avg + 0.05 * ret
            loss = -ret  # maximize
            loss.backward()
            optimizer.step()
        tracked_info.append(running_avg.detach().cpu().numpy())
    y_smoothed = gaussian_filter1d(tracked_info, sigma=5)
    plt.plot(y_smoothed)
    plt.savefig("MI.png")


if __name__ == '__main__':
    # Make sure to use depth config.
    argp = ArgumentParser()
    argp.add_argument('experiment_config')
    cli_args = argp.parse_args()

    yaml_args = yaml.load(open(cli_args.experiment_config))
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    yaml_args['device'] = device
    get_info(yaml_args)
