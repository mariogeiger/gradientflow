# pylint: disable=no-member, invalid-name, line-too-long
"""
Gradient flow for any model using pytorch backprop
"""
import copy
import math

import torch

from .flow import flow
from .gradient import gradient


class _ContinuousMomentum(torch.optim.Optimizer):
    r"""Implements a continuous version of momentum.

    d/dt velocity = -1/tau (velocity + grad)
     or
    d/dt velocity = -mu/t (velocity + grad)

    d/dt parameters = velocity
    """

    def __init__(self, params, dt, tau):
        defaults = dict(dt=dt, tau=tau)
        super().__init__(params, defaults)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable): A closure that reevaluates the model and
                returns the loss. Optional for most optimizers.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            tau = group['tau']
            dt = group['dt']

            for p in group['params']:
                if p.grad is None:
                    continue

                param_state = self.state[p]
                if 't' not in param_state:
                    t = param_state['t'] = 0
                else:
                    t = param_state['t']

                if tau != 0:
                    if 'velocity' not in param_state:
                        v = param_state['velocity'] = torch.zeros_like(p.data)
                    else:
                        v = param_state['velocity']

                if tau > 0:
                    x = math.exp(-dt / tau)
                    v.mul_(x).add_(-(1 - x) * p.grad.data)
                elif tau < 0:
                    mu = -tau
                    x = (t / (t + dt)) ** mu
                    v.mul_(x).add_(-(1 - x) * p.grad.data)
                else:
                    v = -p.grad.data

                p.data.add_(dt * v)
                param_state['t'] += dt

        return loss


def _make_step(f, optimizer, dt, grad):
    """
    internal function
    """
    i = 0
    for p in f.parameters():
        n = p.numel()
        p.grad = grad[i: i + n].view_as(p)
        i += n

    for param_group in optimizer.param_groups:
        param_group['dt'] = dt

    optimizer.step()

    for p in f.parameters():
        p.grad = None


def _output_gradient(f, loss, x, y, out0, chunk):
    """
    internal function
    """
    out = []
    grad = 0
    loss_value = 0
    for i in [slice(i, i + chunk) for i in range(0, len(x), chunk)]:
        o = f(x[i]) - out0[i]
        l = loss(o, y[i])
        assert l.shape == (len(o),)
        l = l.sum() / len(x)
        grad += gradient(l, f.parameters())
        out.append(o)
        loss_value += l.item()
    return torch.cat(out), grad, loss_value


def gradientflow_backprop(f0, x, y, loss, subf0=False, tau=0, chunk=None, batch=None, max_dgrad=1e-3, max_dout=1):
    """
    gradientflow on a torch.nn.Model using backprop
    :param f0: torch.nn.Model
    :param x: torch.Tensor [z, ...]
    :param y: torch.Tensor [z, ...]
    :param loss: function: outputs, labels -> losses
    """

    if chunk is None:
        chunk = len(x)

    if batch is None:
        batch = len(x)

    f = copy.deepcopy(f0)

    with torch.no_grad():
        with torch.no_grad():
            out0 = []
            for i in [slice(i, i + chunk) for i in range(0, len(x), chunk)]:
                out0.append(f0(x[i]))
            out0 = torch.cat(out0)
        if isinstance(subf0, bool):
            if not subf0:
                out0 = torch.zeros_like(out0)
        else:
            assert out0.shape == subf0.shape
            out0 = subf0

    def prepare(sta, t, old_data, old_t):
        sta = copy.deepcopy(sta)
        ff = copy.deepcopy(f)
        ff.load_state_dict(sta[0])

        if old_data is not None:
            if old_t == t:
                if batch == len(x):
                    return old_data
                bi = torch.randperm(len(x))[:batch].sort().values
            else:
                bi = old_data[3]
        else:
            bi = torch.randperm(len(x))[:batch].sort().values

        out, grad, loss_value = _output_gradient(ff, loss, x[bi], y[bi], out0[bi], chunk)

        return out, grad, loss_value, bi

    def make_step(sta, data, _t, dt):
        sta = copy.deepcopy(sta)
        ff = copy.deepcopy(f)
        optimizer = _ContinuousMomentum(ff.parameters(), dt=0, tau=tau)

        ff.load_state_dict(sta[0])
        optimizer.load_state_dict(sta[1])
        _out, grad, _loss_value, _bi = data

        _make_step(ff, optimizer, dt, grad)
        return ff.state_dict(), optimizer.state_dict()

    def compare(data, new_data):
        out, grad, _loss_value, bi = data
        new_out, new_grad, _new_loss_value, new_bi = new_data

        assert bi.eq(new_bi).all()

        dout = (out - new_out).abs().max().item()
        if grad.norm() == 0 or new_grad.norm() == 0:
            dgrad = 0
        else:
            dgrad = (grad - new_grad).norm().pow(2).div(grad.norm() * new_grad.norm()).item()

        return dgrad / max_dgrad, dout / max_dout


    opt = _ContinuousMomentum(f.parameters(), dt=0, tau=tau)

    for state, internals in flow((f.state_dict(), opt.state_dict()), prepare, make_step, compare):
        out, grad, loss_value, bi = internals['data']
        f.load_state_dict(internals['x'][0])

        state = {
            'step': state['step'],
            't': state['t'],
            'loss': loss_value,
            'dt': state['dt'],
            'dgrad': max_dgrad * state['d'][0] if state['d'] is not None else 0,
            'dout': max_dout * state['d'][1] if state['d'] is not None else 0,
        }
        internals = {
            'f': f,
            'output': out,
            'output0': out0[bi],
            'gradient': grad,
            'batch_indices': bi,
            'changed_dt': internals['changed_dt']
        }
        yield state, internals
