"""
Gradient flow for any model using pytorch backprop
"""
import copy
from collections import namedtuple
import itertools

import torch

from .gradient import gradient


def _make_step(f, dt, grad):
    """
    internal function
    """
    f = copy.deepcopy(f)

    with torch.no_grad():
        i = 0
        for p in f.parameters():
            n = p.numel()
            p -= dt * grad[i: i + n].view_as(p)
            i += n

    return f


def _output_gradient(f, loss, x, y, out0, batch_indices, chunk):
    """
    internal function
    """
    x = x[batch_indices]
    y = y[batch_indices]
    if out0 is not None:
        out0 = out0[batch_indices]

    out = []
    grad = 0
    loss_value = 0
    for i in [slice(i, i + chunk) for i in range(0, len(x), chunk)]:
        o = f(x[i])
        if out0 is not None:
            o = o - out0[i]
        l = loss(o, y[i])
        assert l.shape == (len(o),)
        l = l.sum() / len(x)
        grad += gradient(l, f.parameters())
        out.append(o)
        loss_value += l.item()
    return torch.cat(out), grad, loss_value


def _clamp(min_, x, max_):
    return max(min_, min(x, max_))


def dgrad(g1, g2):
    return (g1 - g2).norm().pow(2) / (g1.norm() * g2.norm())


def gradientflow_backprop_sgd(f0, x, y, loss_function, subf0=False, beta=1.0, chunk=None, batch_min=1, batch_max=None, max_dgrad=1e-3, max_dout=1):
    """
    gradientflow

    Parameters
    ----------
    f0 : torch.Module
    x : torch.Tensor
    y : torch.Tensor
    """

    if chunk is None:
        chunk = len(x)

    if batch_max is None:
        batch_max = len(x)

    f = copy.deepcopy(f0)

    if isinstance(subf0, bool):
        if subf0:
            with torch.no_grad():
                out0 = torch.cat([f0(x[i:i + chunk]) for i in range(0, len(x), chunk)])  # comment
        else:
            out0 = None
    else:
        out0 = subf0

    State = namedtuple('State', 'f, t, tx')
    StepData = namedtuple('StepData', 'output, gradient, loss, batch_indices, new_x')

    state = State(f=f, t=0.0, tx=0.0)
    del f

    dt = batch_min / beta
    last_dt = 0
    last_batch = 0
    step_change_dt = 0
    d = (0, 0)

    batch_indices = torch.randperm(len(x))[:batch_min]
    out, grad, loss = _output_gradient(state.f, loss_function, x, y, out0, batch_indices, chunk)
    data = StepData(
        output=out,
        gradient=grad,
        loss=loss,
        batch_indices=batch_indices,
        new_x=batch_min,
    )
    del batch_indices, out, grad, loss

    for step in itertools.count():

        state_ = {
            'step': step,
            't': state.t,
            'tx': state.tx,
            'dt': last_dt,
            'batch': last_batch,
            'loss': data.loss,
            'dout': max_dout * d[0],
            'dgrad': max_dgrad * d[1],
        }
        internals_ = {
            'f': state.f,
            'output': data.output,
            'output0': out0[data.batch_indices] if out0 is not None else None,
            'gradient': data.gradient,
            'batch_indices': data.batch_indices,
            'changed_dt': step_change_dt == step - 1,
        }

        yield state_, internals_

        while True:
            last_dt = dt
            last_batch = data.new_x

            # 2 - Make a tentative step
            new_f = _make_step(state.f, dt, data.gradient)
            new_t = state.t + dt
            new_tx = state.tx + data.new_x / beta
            new_state = State(f=new_f, t=new_t, tx=new_tx)
            del new_f, new_t, new_tx

            # 3 - Check if the step is small enough
            out, grad, loss = _output_gradient(new_state.f, loss_function, x, y, out0, data.batch_indices, chunk)
            new_data = StepData(
                output=out,
                gradient=grad,
                loss=loss,
                batch_indices=data.batch_indices,
                new_x=0
            )
            del out, grad, loss

            d = (
                (data.output - new_data.output).abs().max().item() / max_dout,
                dgrad(data.gradient, new_data.gradient).item() / max_dgrad,
            )
            if all(c < 1 for c in d):
                if all(c < 1/2 for c in d):
                    dt *= 1.1
                break

            # 4 - If not, reset and retry
            dt /= 10
            step_change_dt = step

        # 5 - If yes, compute the new output and gradient
        state = new_state
        del new_state

        batch = round(beta * (state.t + dt - state.tx))
        if batch < batch_min:
            # keep the same batch
            data = new_data
            assert data.new_x == 0
        else:
            if batch > batch_max:
                dt = dt * batch_max / batch
                batch = batch_max
            batch_indices = torch.randperm(len(x))[:batch]

            out, grad, loss = _output_gradient(state.f, loss_function, x, y, out0, batch_indices, chunk)
            data = StepData(
                output=out,
                gradient=grad,
                loss=loss,
                batch_indices=batch_indices,
                new_x=batch,
            )
            del batch_indices, out, grad, loss
        del batch, new_data