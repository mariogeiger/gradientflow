import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from tqdm.auto import tqdm

from gradientflow import gradientflow_backprop_sgd


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.tanh(x)
        x = self.conv2(x)
        x = torch.tanh(x)
        x = F.avg_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = torch.tanh(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
dataset1 = datasets.MNIST(
    'data',
    train=True,
    download=True,
    transform=transform
)

x = torch.stack([x for x, y in dataset1])
y = torch.tensor([y for x, y in dataset1])


def loss(pred, true):
    return torch.nn.functional.nll_loss(pred, true, reduction='none')


f = Net()
dynamics = []

stop = 2_000
with tqdm(total=stop) as pb:
    for state, internals in gradientflow_backprop_sgd(
                    f,
                    x,
                    y,
                    loss,
                    beta=2e5,
                    batch_min=10,
                    batch_max=1000,
                    max_dout=1e-1,
                    max_dgrad=1e-3,
                    chunk=200,
                                                     ):

        pb.update(1)
        dynamics.append(state)
        if state['step'] == stop:
            break

plt.plot([x['t'] for x in dynamics], [x['loss'] for x in dynamics])
plt.show()
