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


def _output_gradient(f, loss_function, dataset, labels, out0, batch_indices, chunk):
    """
    internal function
    """
    try:
        x = dataset[batch_indices]
    except TypeError:
        x = [dataset[i.item()] for i in batch_indices]

    try:
        y = labels[batch_indices]
    except TypeError:
        y = [labels[i.item()] for i in batch_indices]

    if out0 is not None:
        out0 = out0[batch_indices]

    out = []
    grad = 0
    loss_value = 0
    for i in [slice(i, i + chunk) for i in range(0, len(x), chunk)]:
        o = f(x[i])
        if out0 is not None:
            o = o - out0[i]
        l = loss_function(o, y[i])
        assert l.shape == (len(o),)
        l = l.sum() / len(x)
        grad += gradient(l, f.parameters())
        out.append(o)
        loss_value += l.item()
    return torch.cat(out), grad, loss_value


def _clamp(min_, x, max_):
    return max(min_, min(x, max_))


def _dgrad(g1, g2):
    return (g1 - g2).norm().pow(2) / (g1.norm() * g2.norm())


_State = namedtuple('State', 'f, t, tx')
_StepData = namedtuple('StepData', 'output, gradient, loss, batch_indices, new_x')


def _prepare_step(state, post_last_step_data, dt, dataset, labels, loss_function, out0, beta, chunk, batch_min, batch_max):
    batch = round(beta * (state.t + dt - state.tx))
    if batch < batch_min:
        if post_last_step_data.new_x == 0:
            # keep the same batch if the last (tentative) step was a success
            return post_last_step_data

        batch = batch_min

    if batch > batch_max:
        dt = dt * batch_max / batch
        batch = batch_max

    batch_indices = torch.randperm(len(dataset))[:batch]

    out, grad, loss = _output_gradient(state.f, loss_function, dataset, labels, out0, batch_indices, chunk)
    return _StepData(
        output=out,
        gradient=grad,
        loss=loss,
        batch_indices=batch_indices,
        new_x=batch,
    )


def gradientflow_backprop_sgd(f0, dataset, labels, loss_function, subf0=False, beta=1.0, chunk=None, batch_min=1, batch_max=None, max_dgrad=1e-3, max_dout=1):
    """
    gradientflow

    Parameters
    ----------
    f0 : torch.Module
    dataset : any
    labels : torch.Tensor
    """

    if chunk is None:
        chunk = len(dataset)

    if batch_max is None:
        batch_max = len(dataset)

    f = copy.deepcopy(f0)

    if isinstance(subf0, bool):
        if subf0:
            with torch.no_grad():
                out0 = torch.cat([f0(dataset[i:i + chunk]) for i in range(0, len(dataset), chunk)])  # comment
        else:
            out0 = None
    else:
        out0 = subf0

    state = _State(f=f, t=0.0, tx=0.0)
    del f

    dt = batch_max / beta
    last_dt = 0
    last_batch = 0
    step_change_dt = 0
    d = (0, 0)

    batch_indices = torch.randperm(len(dataset))[:batch_max]
    out, grad, loss = _output_gradient(state.f, loss_function, dataset, labels, out0, batch_indices, chunk)
    data = _StepData(
        output=out,
        gradient=grad,
        loss=loss,
        batch_indices=batch_indices,
        new_x=batch_max,
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
            new_state = _State(f=new_f, t=new_t, tx=new_tx)
            del new_f, new_t, new_tx

            # 3 - Check if the step is small enough
            out, grad, loss = _output_gradient(new_state.f, loss_function, dataset, labels, out0, data.batch_indices, chunk)
            post_step_data = _StepData(
                output=out,
                gradient=grad,
                loss=loss,
                batch_indices=data.batch_indices,
                new_x=data.new_x
            )
            del out, grad, loss

            d = (
                (data.output - post_step_data.output).abs().max().item() / max_dout,
                _dgrad(data.gradient, post_step_data.gradient).item() / max_dgrad,
            )
            if all(c < 1 for c in d):
                if all(c < 1/2 for c in d):
                    dt *= 1.1

                # success!
                post_step_data = _StepData(
                    output=post_step_data.output,
                    gradient=post_step_data.gradient,
                    loss=post_step_data.loss,
                    batch_indices=post_step_data.batch_indices,
                    new_x=0  # if reused, marked as not new
                )
                break

            # 4 - If not, reset and retry
            dt /= 1.1**3  # = 1.33
            step_change_dt = step

            data = _prepare_step(state, post_step_data, dt, dataset, labels, loss_function, out0, beta, chunk, batch_min, batch_max)

        # 5 - If yes, compute the new output and gradient
        state = new_state
        del new_state

        data = _prepare_step(state, post_step_data, dt, dataset, labels, loss_function, out0, beta, chunk, batch_min, batch_max)
