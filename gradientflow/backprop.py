# pylint: disable=no-member, invalid-name, line-too-long
"""
Gradient flow for any model using pytorch backprop
"""
import copy
import itertools
import math

import torch

from .gradient import gradient


class ContinuousMomentum(torch.optim.Optimizer):
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


def make_step(f, optimizer, dt, grad):
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


def output_gradient(f, loss, x, y, out0, chunk):
    """
    internal function
    """
    out = []
    grad = 0
    loss_value = 0
    for i in [slice(i, i + chunk) for i in range(0, len(x), chunk)]:
        o = f(x[i]) - out0[i]
        l = loss(o, y[i]).sum() / len(x)
        grad += gradient(l, f.parameters())
        out.append(o)
        loss_value += l.item()
    return torch.cat(out), grad, loss_value


def gradientflow_backprop(f0, x, y, loss, subf0=False, tau=0, chunk=None, batch=None, max_dgrad=1e-3, max_dout=math.inf):
    """
    gradientflow on a torch.nn.Model using backprop
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

    dt = 1
    current_dt = 0
    step_change_dt = 0
    optimizer = ContinuousMomentum(f.parameters(), dt=dt, tau=tau)

    t = 0

    bi = torch.randperm(len(x))[:batch].sort().values
    out, grad, loss_value = output_gradient(f, loss, x[bi], y[bi], out0[bi], chunk)
    dgrad, dout = 0, 0

    for step in itertools.count():

        state = {
            'step': step,
            't': t,
            'loss': loss_value,
            'dt': current_dt,
            'dgrad': dgrad,
            'dout': dout,
            'changed_dt': step_change_dt == step - 1
        }
        internals = {
            'f': f,
            'output': out,
            'output0': out0[bi],
            'gradient': grad,
            'batch_indices': bi,
        }

        yield state, internals

        if torch.isnan(out).any():
            break

        # 1 - Save current state
        state = copy.deepcopy((f.state_dict(), optimizer.state_dict(), t))

        while True:
            # 2 - Make a tentative step
            make_step(f, optimizer, dt, grad)
            t += dt
            current_dt = dt

            # 3 - Check if the step is small enough
            new_out, new_grad, new_loss_value = output_gradient(f, loss, x[bi], y[bi], out0[bi], chunk)

            if torch.isnan(new_out).any():
                break

            dout = (out - new_out).abs().max().item()
            if grad.norm() == 0 or new_grad.norm() == 0:
                dgrad = 0
            else:
                dgrad = (grad - new_grad).norm().pow(2).div(grad.norm() * new_grad.norm()).item()

            if dgrad < max_dgrad and dout < max_dout:
                if dgrad < 0.5 * max_dgrad and dout < 0.5 * max_dout:
                    dt *= 1.1
                break

            # 4 - If not, reset and retry
            dt /= 10

            # print("[{} +{}] [dt={:.1e} dgrad={:.1e} dout={:.1e}]".format(step, step - step_change_dt, dt, dgrad, dout), flush=True)
            step_change_dt = step
            f.load_state_dict(state[0])
            optimizer.load_state_dict(state[1])
            t = state[2]

        # 5 - If yes, compute the new output and gradient
        if batch == len(x):
            out = new_out
            grad = new_grad
            loss_value = new_loss_value
        else:
            bi = torch.randperm(len(x))[:batch].sort().values
            out, grad, loss_value = output_gradient(f, loss, x[bi], y[bi], out0[bi], chunk)
