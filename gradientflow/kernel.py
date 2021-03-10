# pylint: disable=no-member, invalid-name, line-too-long
"""
Gradient flow for any model using pytorch backprop
"""
import math
from collections import namedtuple

from .flow import flow


def gradientflow_kernel(kernel, y, loss_prim, f0=None, tau=0, max_dgrad=1e-3, max_dout=1):
    """gradientflow for a kernel

    model: out = f0 + kernel @ alpha

    Parameters
    ==========
    kernel : torch.Tensor
        tensor of shape ``(p, p)``

    y : torch.Tensor
        the labels, shape ``(p, ...)``

    loss_prim : function
        loss derivative
        signature: (output: tensor of shape (p,), labels: tensor of shape (p, ...)) -> tensor of shape (p,)

    f0 : torch.Tensor, optional
        ``(p,)`` shape tensor

    tau : float
        momentum parameter

    max_dgrad : float
        constraint for the adaptative time step

    max_dout : float
        constraint for the adaptative time step
    """
    if f0 is None:
        f0 = kernel.new_zeros(len(y))

    State = namedtuple('State', 'alpha, velo')
    Data = namedtuple('Data', 'output, gradient')

    def prepare(sta: State, t: float, old_data: Data, old_t: float) -> Data:
        out = f0 + kernel @ sta.alpha
        grad = loss_prim(out, y) / len(y)
        return Data(out, grad)

    def make_step(sta: State, data: Data, t: float, dt: float) -> State:
        alpha = sta.alpha.clone()
        velo = sta.velo.clone()
        grad = data.gradient

        if tau > 0:
            x = math.exp(-dt / tau)
            velo.mul_(x).add_(-(1 - x) * grad)
        elif tau < 0:
            mu = -tau
            x = (t / (t + dt)) ** mu
            velo.mul_(x).add_(-(1 - x) * grad)
        else:
            velo.copy_(-grad)
        alpha.add_(dt * velo)
        return State(alpha, velo)

    def compare(data: Data, new_data: Data) -> float:
        out, grad = data
        new_out, new_grad = new_data

        dout = (out - new_out).abs().max().item()
        if grad.norm() == 0 or new_grad.norm() == 0:
            dgrad = 0
        else:
            dgrad = (grad - new_grad).norm().pow(2).div(grad.norm() * new_grad.norm()).item()

        return dgrad / max_dgrad, dout / max_dout

    x0 = State(kernel.new_zeros(len(y)), kernel.new_zeros(len(y)))

    for state, internals in flow(x0, prepare, make_step, compare):
        out, grad = internals['data']
        alpha, velo = internals['x']

        state = {
            'step': state['step'],
            't': state['t'],
            'dt': state['dt'],
            'dgrad': max_dgrad * state['d'][0] if state['d'] is not None else 0,
            'dout': max_dout * state['d'][1] if state['d'] is not None else 0,
        }
        internals = {
            'alpha': alpha,
            'velo': velo,
            'output': out,
            'gradient': grad,
            'changed_dt': internals['changed_dt']
        }
        yield state, internals
