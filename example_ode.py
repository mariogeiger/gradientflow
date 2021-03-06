# pylint: disable=no-member, invalid-name, line-too-long, not-callable
"""
pendulum equation
"""
import torch
import matplotlib.pyplot as plt

from gradientflow import flow_ode


theta0 = 3.14 / 2
dot_theta0 = 0.0


def grad(x, _t):
    """
    :param x: position of the pendulum
    :param t: time
    """
    theta, dot_theta = x
    return torch.tensor([dot_theta, -theta.sin()])


dynamics = []

for state, internals in flow_ode(torch.tensor([theta0, dot_theta0]), grad, max_dgrad=1e-5):
    state['theta'] = internals['x'][0].item()
    state['dot_theta'] = internals['x'][1].item()
    dynamics.append(state)

    print(state)

    if state['t'] > 100:
        break

plt.plot([x['t'] for x in dynamics], [x['theta'] for x in dynamics])
plt.show()
