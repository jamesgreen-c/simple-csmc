import argparse
import os

import matplotlib.pyplot as plt
import numpy as np

from src.kernels import KernelType
from src.utils.printing import ctext


# ARGS PARSING
parser = argparse.ArgumentParser()

parser.add_argument("--T", dest="T", type=int, default=100)
parser.add_argument("--D", dest="D", type=int, default=1)
parser.add_argument("--log-var", dest="log_var", type=float, default=0)
parser.add_argument("--kernel", dest="kernel", type=int, default=KernelType.CSMC)
parser.add_argument("--style", dest="style", type=str, default="bootstrap")
parser.add_argument("--N", dest="N", type=int, default=31)  # total number of particles is N + 1
parser.add_argument("--seed", dest="seed", type=int, default=1234)

parser.add_argument("--i", type=int, default=0)

args = parser.parse_args()

# CONFIG
kernel_type = KernelType(args.kernel)
Ts = np.arange(args.T)

# functions
def plot_means(data, dirpath, dim: int = 0):
    """
    Plot the resulting smoothing samples for experiment i. 
    Compare this to the means from a Kalman smoother, and the ground truth.
    
    data:
        smoothing_means: (K, T, D)
        true_xs:         (K, T, D)
        ys:              (K, T, D)
        samples:         (K, M, T, D)
    
    args.i: The experiment index (which experiment to view)
    """

    s_ms = data["smoothing_means"][args.i]
    true_xs = data["true_xs"][args.i]
    ys = data["ys"][args.i]
    samples = data["samples"][args.i]

    M, T, D = samples.shape
    M = min(M, 10)  # plot maximum of 10 chains

    fig, ax = plt.subplots(M, 1, figsize=(15, 5*M))
    ax = np.atleast_1d(ax)
    for m in range(M):
       ax[m].plot(Ts, true_xs[:, dim], color="black", label="Truth") 
       ax[m].plot(Ts, s_ms[:, dim], color="red", linestyle="--", label="Kalman mean") 
       ax[m].plot(Ts, samples[m, :, dim], color="blue", linestyle="--", label="Particle mean")
       ax[m].scatter(Ts, ys[:, dim], color="green", marker="x", label="observations")

       ax[m].set_title(f"Chain: {m}, Dimension: {dim}")
       ax[m].set_xlabel("Time")
       ax[m].legend()

    plt.tight_layout()
    fig.savefig(f"{dirpath}/samples.png", dpi=200)
    plt.close()


def load_data():
    experiment_name = "kernel={},style={},T={},D={},N={},log_var={},seed={}"
    experiment_name = experiment_name.format(
        kernel_type.name, 
        args.style, 
        args.T, 
        args.D, 
        args.N, 
        args.log_var,
        args.seed,
    )
    dirpath = f"results/{experiment_name}"
    if not os.path.exists(dirpath):
        print(ctext("No such experiment exists", "yellow"))
        print(experiment_name)
        exit()

    data = np.load(f"{dirpath}/data.npz")
    return data, dirpath


# run plotting
data, dirpath = load_data()
plot_means(data, dirpath)
