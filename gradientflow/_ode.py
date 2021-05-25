# pylint: disable=no-member, invalid-name, line-too-long
"""
flow for an ODE
"""
from ._flow import flow


def flow_ode(x, grad_fn, max_dgrad=1e-4):
    """
    flow for an ODE
    """

    def prepare(xx, t, old_data, old_t):
        if old_data is not None and old_t == t:
            return old_data
        return grad_fn(xx, t)

    def make_step(xx, g, _t, dt):
        return xx + dt * g

    def compare(g1, g2):
        dgrad = (g1 - g2).pow(2).sum() / (g1.pow(2).sum() * g2.pow(2).sum()).sqrt()
        return dgrad.item() / max_dgrad

    for state, internals in flow(x, prepare, make_step, compare):
        yield state, internals
