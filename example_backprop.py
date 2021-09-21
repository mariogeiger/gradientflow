# pylint: disable=no-member, invalid-name, line-too-long
"""
Perceptron trained with gradientflow
"""
import torch
import matplotlib.pyplot as plt

from gradientflow import gradientflow_backprop

n = 50
d = 100
f = torch.nn.Linear(d, 1)
x = torch.randn(n, d)
y = torch.randn(n)


def loss(pred, true):
    """
    :param pred: output of the model `f`
    :param true: the labels `y`
    """
    return (pred.flatten() - true).pow(2)


def checkpoints():
    t = 0.1
    while True:
        yield t
        t *= 1.1


dynamics = []

for state, internals in gradientflow_backprop(f, x, y, loss, checkpoints=checkpoints()):
    dynamics.append(state)

    if state['loss'] < 1e-5:
        f1 = internals['f']  # trained model
        break

plt.plot([x['t'] for x in dynamics], [x['loss'] for x in dynamics])
plt.xscale('log')
plt.yscale('log')
plt.show()
