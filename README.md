# gradientflow

## Install
```
pip install git+https://github.com/mariogeiger/gradientflow
```

## Usage
```python
import torch
import matplotlib.pyplot as plt

from gradientflow import gradientflow_backprop

n = 50
d = 100
f = torch.nn.Linear(d, 1)
x = torch.randn(n, d)
y = torch.randn(n)

def loss(pred, true):
    return (pred.flatten() - true).pow(2)

dynamics = []

for state, internals in gradientflow_backprop(f, x, y, loss):
    dynamics.append(state)

    if state['loss'] < 1e-5:
        break

f1 = internals['f']  # trained model

plt.plot([x['t'] for x in dynamics], [x['loss'] for x in dynamics])
plt.yscale('log')
```
