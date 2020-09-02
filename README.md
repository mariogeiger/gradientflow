# Gradient Flow

Implementation of gradient flow in pytorch.
```
dw/dt = -dL/dw
```

## Install
```
pip install git+https://github.com/mariogeiger/gradientflow
```

## Usage
```python
f = torch.nn.Linear(50, 1)
x = torch.randn(100, 50)
y = torch.randn(100)

def loss(pred, true): return (pred.flatten() - true).pow(2)

dynamics = []

for state, internals in gradientflow_backprop(f, x, y, loss):
    dynamics.append(state)

    if state['loss'] < 1e-5:
        break

plt.plot([x['t'] for x in dynamics], [x['loss'] for x in dynamics])
```
(subset of [example.py](example.py))

![image](https://user-images.githubusercontent.com/333780/91983505-141cd800-ed2c-11ea-8a3c-80f436ffada3.png)

