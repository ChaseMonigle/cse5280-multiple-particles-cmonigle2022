#!/usr/bin/env python3
"""
CSE5280 - Multi-Particle Animation with Social Forces
Multi-particle gradient descent with:
- Goal attraction
- Wall penalty
- Social force penalties:
  (1) Isotropic quadratic repulsion (personal space)
  (2) Isotropic exponential repulsion
  (3) Anisotropic (velocity-dependent) exponential repulsion

Run:
  python3 multi_particle_social_forces.py --model iso_quad
  python3 multi_particle_social_forces.py --model iso_exp
  python3 multi_particle_social_forces.py --model anisotropic

Optional:
  python3 multi_particle_social_forces.py --model anisotropic --save mp4 --outfile demo.mp4
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


# ----------------------------
# helpers
# ----------------------------
def safe_norm(v: np.ndarray, eps: float = 1e-9) -> float:
    n = float(np.linalg.norm(v))
    return n if n > eps else eps


# ----------------------------
# cost gradients
# ----------------------------
def goal_gradient(x: np.ndarray, goal: np.ndarray) -> np.ndarray:
    """
    Quadratic goal cost: 0.5 * ||x - goal||^2
    grad = x - goal
    """
    return x - goal


def wall_gradient_box(x: np.ndarray, box_min: float, box_max: float, margin: float) -> np.ndarray:
    """
    Soft quadratic penalty near the boundary of a square box [box_min, box_max]^2.
    Only applies when inside a 'margin' band close to a wall.
    """
    gx, gy = 0.0, 0.0

    # left wall band
    if x[0] < box_min + margin:
        gx += x[0] - (box_min + margin)

    # right wall band
    if x[0] > box_max - margin:
        gx += x[0] - (box_max - margin)

    # bottom wall band
    if x[1] < box_min + margin:
        gy += x[1] - (box_min + margin)

    # top wall band
    if x[1] > box_max - margin:
        gy += x[1] - (box_max - margin)

    return np.array([gx, gy], dtype=float)


def social_grad_iso_quadratic(i: int, X: np.ndarray, R: float, eps: float = 1e-9) -> np.ndarray:
    """
    Isotropic quadratic repulsion ("personal space"):
      C = 0.5 * (R - d)^2 for d <= R, else 0
    grad wrt xi:
      (R - d) * (xi - xj)/d   for d <= R
    """
    grad = np.zeros(2, dtype=float)
    xi = X[i]

    for j in range(X.shape[0]):
        if j == i:
            continue

        diff = xi - X[j]
        d = float(np.linalg.norm(diff))

        if d <= R and d > eps:
            grad += (R - d) * (diff / d)

    return grad


def social_grad_iso_exponential(i: int, X: np.ndarray, A: float, B: float, eps: float = 1e-9) -> np.ndarray:
    """
    Isotropic exponential social force:
      C = A * exp(-d/B)
    grad wrt xi:
      (A/B) * exp(-d/B) * (xi - xj)/d
    """
    grad = np.zeros(2, dtype=float)
    xi = X[i]

    for j in range(X.shape[0]):
        if j == i:
            continue

        diff = xi - X[j]
        d = float(np.linalg.norm(diff))

        if d > eps:
            grad += (A / B) * np.exp(-d / B) * (diff / d)

    return grad


def social_grad_anisotropic_exp(
    i: int,
    X: np.ndarray,
    V: np.ndarray,
    A: float,
    B: float,
    beta: float,
    eps: float = 1e-9
) -> np.ndarray:
    """
    Anisotropic (velocity-dependent) exponential interaction:
      C_ani(i,j) = (1 + beta * max(0, vhat_i · (xj-xi)/||xj-xi||)) * phi(d)
    where phi(d) is the same distance penalty (here exponential).
    We apply the same weight to the gradient of phi(d) wrt xi.

    Note:
    - When speed is ~0, vhat is treated as 0-vector -> anisotropic weight becomes 1.
    """
    grad = np.zeros(2, dtype=float)
    xi = X[i]

    vi = V[i]
    speed = float(np.linalg.norm(vi))
    if speed > eps:
        vhat = vi / speed
    else:
        vhat = np.zeros(2, dtype=float)

    for j in range(X.shape[0]):
        if j == i:
            continue

        xj = X[j]
        diff = xi - xj
        d = float(np.linalg.norm(diff))

        if d > eps:
            dhat_forward = (xj - xi) / d  # direction from i to j
            directional = max(0.0, float(np.dot(vhat, dhat_forward)))
            weight = 1.0 + beta * directional

            # base exponential gradient (same as isotropic)
            grad += weight * (A / B) * np.exp(-d / B) * (diff / d)

    return grad


# ----------------------------
# simulation
# ----------------------------
def simulate(
    model: str,
    N: int,
    T: int,
    alpha: float,
    seed: int,
    box_min: float,
    box_max: float,
    wall_margin: float,
    goal: np.ndarray,
    R: float,
    A: float,
    B: float,
    beta: float,
) -> tuple[list[np.ndarray], np.ndarray]:
    """
    Returns:
      trajectory: list of (N,2) positions over time
      goal: goal position
    """
    rng = np.random.default_rng(seed)

    # start positions in a smaller region (so they have room to move)
    X = rng.uniform(low=box_min + 1.0, high=box_min + 4.0, size=(N, 2))
    V = np.zeros_like(X)

    trajectory = []

    for _t in range(T):
        X_new = X.copy()

        for i in range(N):
            grad = np.zeros(2, dtype=float)

            # goal + walls
            grad += goal_gradient(X[i], goal)
            grad += wall_gradient_box(X[i], box_min, box_max, wall_margin)

            # social
            if model == "iso_quad":
                grad += social_grad_iso_quadratic(i, X, R)
            elif model == "iso_exp":
                grad += social_grad_iso_exponential(i, X, A, B)
            elif model == "anisotropic":
                grad += social_grad_anisotropic_exp(i, X, V, A, B, beta)
            else:
                raise ValueError(f"Unknown model: {model}")

            # gradient descent update
            X_new[i] = X[i] - alpha * grad

            # keep inside the box (hard clamp just so nothing flies away)
            X_new[i, 0] = np.clip(X_new[i, 0], box_min, box_max)
            X_new[i, 1] = np.clip(X_new[i, 1], box_min, box_max)

        V = X_new - X
        X = X_new
        trajectory.append(X.copy())

    return trajectory, goal


# ----------------------------
# visualization
# ----------------------------
from typing import Optional, List

def animate(
    trajectory,
    goal,
    box_min,
    box_max,
    title,
    save=None,
    outfile="animation.mp4",
    fps=30
):
    fig, ax = plt.subplots()
    ax.set_xlim(box_min, box_max)
    ax.set_ylim(box_min, box_max)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title)

    # draw goal
    ax.scatter([goal[0]], [goal[1]], marker="*", s=160)

    # initial scatter
    scat = ax.scatter([], [], s=40)

    def init():
        scat.set_offsets(np.zeros((0, 2)))
        return (scat,)

    def update(frame):
        X = trajectory[frame]
        scat.set_offsets(X)
        return (scat,)

    ani = FuncAnimation(
        fig,
        update,
        frames=len(trajectory),
        init_func=init,
        interval=1000 / fps,
        blit=True
    )

    if save is not None:
        if save.lower() == "gif":
            ani.save(outfile, writer="pillow", fps=fps)
        elif save.lower() == "mp4":
            ani.save(outfile, writer="ffmpeg", fps=fps)
        else:
            raise ValueError("save must be 'gif' or 'mp4' (or omit --save).")

    plt.show()


# ----------------------------
# main
# ----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="anisotropic",
                        choices=["iso_quad", "iso_exp", "anisotropic"],
                        help="Which social force model to run.")
    parser.add_argument("--N", type=int, default=15, help="Number of particles.")
    parser.add_argument("--T", type=int, default=400, help="Number of iterations.")
    parser.add_argument("--alpha", type=float, default=0.05, help="Step size.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed.")

    # environment
    parser.add_argument("--box_min", type=float, default=0.0)
    parser.add_argument("--box_max", type=float, default=10.0)
    parser.add_argument("--wall_margin", type=float, default=0.6)

    # goal
    parser.add_argument("--goalx", type=float, default=8.0)
    parser.add_argument("--goaly", type=float, default=8.0)

    # social params
    parser.add_argument("--R", type=float, default=0.6, help="Personal space radius for iso_quad.")
    parser.add_argument("--A", type=float, default=1.0, help="Exponential strength.")
    parser.add_argument("--B", type=float, default=0.6, help="Exponential decay length.")
    parser.add_argument("--beta", type=float, default=2.0, help="Directional bias strength for anisotropic.")

    # saving
    parser.add_argument("--save", type=str, default=None, choices=[None, "gif", "mp4"],
                        help="Optionally save animation as gif or mp4.")
    parser.add_argument("--outfile", type=str, default="animation.mp4")
    parser.add_argument("--fps", type=int, default=30)

    args = parser.parse_args()

    goal = np.array([args.goalx, args.goaly], dtype=float)

    traj, goal = simulate(
        model=args.model,
        N=args.N,
        T=args.T,
        alpha=args.alpha,
        seed=args.seed,
        box_min=args.box_min,
        box_max=args.box_max,
        wall_margin=args.wall_margin,
        goal=goal,
        R=args.R,
        A=args.A,
        B=args.B,
        beta=args.beta,
    )

    title = f"Multi-Particle Social Forces ({args.model})"
    animate(
        trajectory=traj,
        goal=goal,
        box_min=args.box_min,
        box_max=args.box_max,
        title=title,
        save=args.save,
        outfile=args.outfile,
        fps=args.fps
    )


if __name__ == "__main__":
    main()
