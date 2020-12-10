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
(subset of [example_backprop.py](example_backprop.py))

![image](https://user-images.githubusercontent.com/333780/91983505-141cd800-ed2c-11ea-8a3c-80f436ffada3.png)


```python
theta0 = 3.14 / 2
dot_theta0 = 0.0

def grad(x):
    theta, dot_theta = x
    return torch.tensor([dot_theta, -theta.sin()])

dynamics = []

for state, internals in flow_ode(torch.tensor([theta0, dot_theta0]), grad):
    state['theta'] = internals['variables'][0].item()
    dynamics.append(state)

    if state['t'] > 100:
        break

plt.plot([x['t'] for x in dynamics], [x['theta'] for x in dynamics])
```
(subset of [example_ode.py](example_ode.py))
