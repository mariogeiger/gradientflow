# pylint: disable=no-member, invalid-name, line-too-long
"""
Gradient flow for a kernel method
"""
import copy
import itertools
import math

import torch


def gradientflow_kernel(ktrtr, ytr, tau, loss_prim, max_dgrad=math.inf, max_dout=math.inf):
    """
    gradientflow on a kernel
    """

    alpha = ktrtr.new_zeros(len(ytr))
    velo = alpha.clone()

    dt = 1
    step_change_dt = 0

    t = 0
    current_dt = 0

    otr = ktrtr @ alpha
    grad = loss_prim(otr, ytr) / len(ytr)
    dgrad, dout = 0, 0

    for step in itertools.count():

        state = {
            'step': step,
            't': t,
            'dt': current_dt,
            'dgrad': dgrad,
            'dout': dout,
            'changed_dt': step_change_dt == step - 1
        }
        internals = {
            'output': otr,
            'parameters': alpha,
            'velocity': velo,
            'gradient': grad,
        }

        yield state, internals

        if torch.isnan(alpha).any():
            break

        # 1 - Save current state
        state = copy.deepcopy((alpha, velo, t))

        while True:
            # 2 - Make a tentative step
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

            t += dt
            current_dt = dt

            # 3 - Check if the step is small enough
            otr = ktrtr @ alpha
            new_grad = loss_prim(otr, ytr) / len(ytr)

            dout = (dt * ktrtr @ velo).abs().max().item()
            if grad.norm() == 0 or new_grad.norm() == 0:
                dgrad = 0
            else:
                dgrad = (grad - new_grad).norm().pow(2).div(grad.norm() * new_grad.norm()).item()

            if dgrad < max_dgrad and dout < max_dout:
                if dgrad < 0.1 * max_dgrad and dout < 0.1 * max_dout:
                    dt *= 1.1
                break

            # 4 - If not, reset and retry
            dt /= 10

            # print("[{} +{}] [dt={:.1e} dgrad={:.1e} dout={:.1e}]".format(step, step - step_change_dt, dt, dgrad, dout), flush=True)
            step_change_dt = step
            alpha.copy_(state[0])
            velo.copy_(state[1])
            t = state[2]

        # 5 - If yes, compute the new output and gradient
        grad = new_grad
