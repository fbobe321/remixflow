"""Flow-matching SDEdit loop — the glue that ties the ported components into a
generation. Pure array math (no weights): build the truncated sigma schedule,
noise the source latents to the start sigma, then Euler-integrate the DiT
velocity down to 0. Mirrors RemixFlow's ACE-Step backend + the diffusers
FlowMatchEulerDiscreteScheduler.

    python sdedit.py     # validates the schedule + Euler step vs diffusers
"""
from __future__ import annotations

import numpy as np


def sigma_schedule(strength, steps=8, shift=3.0):
    """`steps` sigmas spanning [strength, 0], flow-matching shifted. Same as the
    RemixFlow ACE-Step backend's SDEdit schedule."""
    base = np.linspace(strength, 0.0, steps + 1)[:-1]
    return shift * base / (1 + (shift - 1) * base)


def noise_init(src_latents, sigma0, noise):
    """Flow-matching interpolation x = (1-σ)·x0 + σ·ε."""
    return (1.0 - sigma0) * src_latents + sigma0 * noise


def euler_loop(x, sched, dit_fn):
    """Integrate dx/dt = v from σ0 down to 0.

    dit_fn(x, sigma) -> velocity v (calls the ported DiT with timestep=sigma).
    Euler: x += (σ_next − σ)·v ; terminal σ_next = 0.
    """
    sched = list(sched)
    for i, sigma in enumerate(sched):
        v = dit_fn(x, sigma)
        sigma_next = sched[i + 1] if i + 1 < len(sched) else 0.0
        x = x + (sigma_next - sigma) * v
    return x


def _validate():
    """Confirm sigma_schedule + Euler step reproduce diffusers' scheduler exactly
    (on a synthetic velocity — the DiT itself is validated separately)."""
    import torch
    from diffusers import FlowMatchEulerDiscreteScheduler

    strength, steps, shift = 0.6, 8, 3.0
    sched = sigma_schedule(strength, steps, shift)

    sch = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1, shift=1.0)
    sch.set_timesteps(sigmas=sched.tolist(), device="cpu")

    rng = np.random.default_rng(0)
    x0 = rng.standard_normal((1, 20, 64)).astype(np.float32)

    # ours
    xo = x0.copy()
    vs = [rng.standard_normal(x0.shape).astype(np.float32) for _ in range(len(sched))]
    for i, sigma in enumerate(sched):
        sn = sched[i + 1] if i + 1 < len(sched) else 0.0
        xo = xo + (sn - sigma) * vs[i]

    # diffusers
    xt = torch.tensor(x0)
    for i, t in enumerate(sch.timesteps):
        xt = sch.step(torch.tensor(vs[i]), t, xt, return_dict=False)[0]
    xt = xt.numpy()

    err = np.abs(xo - xt).max() / np.abs(xt).max()
    print("schedule:", np.round(sched, 4))
    print(f"Euler-loop vs diffusers scheduler: rel max err {err:.3e}")
    print("FLOW_LOOP_PARITY_PASS" if err < 1e-5 else "FAIL")


if __name__ == "__main__":
    _validate()
