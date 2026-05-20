import argparse
import os

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import tqdm

from src.kernels import KernelType
from src.model import get_data
from src.utils.common import barker_move
from src.utils.kalman import smoothing, filtering
from src.utils.resamplings import multinomial

# jax.config.update("jax_enable_x64", False)
# jax.config.update("jax_platform_name", "cpu")

# ARGS PARSING
parser = argparse.ArgumentParser()

parser.add_argument("--T", dest="T", type=int, default=100)
parser.add_argument("--D", dest="D", type=int, default=1)
parser.add_argument("--K", dest="K", type=int, default=1)
parser.add_argument("--M", dest="M", type=int, default=1)
parser.add_argument("--log-var", dest="log_var", type=float, default=0)

parser.add_argument("--kernel", dest="kernel", type=int, default=KernelType.CSMC)
parser.add_argument("--style", dest="style", type=str, default="bootstrap")

parser.add_argument("--backward", action='store_true')
parser.add_argument('--no-backward', dest='backward', action='store_false')
parser.set_defaults(backward=True)

parser.add_argument("--N", dest="N", type=int, default=31)  # total number of particles is N + 1
parser.add_argument("--seed", dest="seed", type=int, default=1234)

args = parser.parse_args()

print(f"""
###############################
#        LGSSM EXAMPLE        #
###############################
Configuration:
    - T: {args.T}
    - kernel: {KernelType(args.kernel).name}
    - style: {args.style}
    - D: {args.D}
""")


# CONFIG
KEY = jax.random.PRNGKey(args.seed)
EXPERIMENT_KEYS = jax.random.split(KEY, args.K)

kernel_type = KernelType(args.kernel)
SIGMA = 10 ** (args.log_var / 2)

def get_smoothing_dist(ys):

    m0 = jnp.zeros((args.D,))
    P0 = H = R = jnp.eye(args.D)
    Hs = jnp.repeat(H[None, ...], args.T, axis=0)
    Rs = jnp.repeat(R[None, ...], args.T, axis=0)
    cs = jnp.zeros((args.T, args.D))
    Fs, Qs, bs = Hs[1:], Rs[1:], cs[1:]

    P0 = SIGMA ** 2 * P0
    Qs = SIGMA ** 2 * Qs
    fms, fPs, _ = filtering.filtering(ys, m0, P0, Fs, Qs, bs, Hs, Rs, cs)
    smoothing_means, smoothing_covs = smoothing.smoothing(fms, fPs, Fs, Qs, bs)
    return smoothing_means, smoothing_covs


@jax.jit
def one_experiment(key):
    data_key, sample_key = jax.random.split(key, 2)
    true_xs, ys = get_data(data_key, SIGMA, args.D, args.T)

    # Kernel construction
    kernel_, init, = kernel_type.kernel_maker(ys, SIGMA, N=args.N,
                                              resampling_func=multinomial,
                                              backward=args.backward,
                                              ancestor_move_func=barker_move,
                                              style=args.style,
                                              conditional=False,
                                              )
    kernel_ = jax.jit(kernel_)

    # We get the means from a Kalman smoother for comparison
    s_ms, s_covs = get_smoothing_dist(ys)

    # init_state will have no effect if conditional = False, so we can pass Kalman smoothing means for shape with no effect
    init_state = init(s_ms)
    def _smoothing_sample(k_):
        sample, *_ = kernel_(k_, init_state)
        return sample

    # vectorises over args.M chains
    sample_keys = jax.random.split(sample_key, args.M)
    samples = jax.vmap(_smoothing_sample)(sample_keys)
    return samples, s_ms, s_covs, true_xs, ys


samples = np.empty((args.K, args.M, args.T, args.D))
means = np.empty((args.K, args.T, args.D))
covs = np.empty((args.K, args.T, args.D, args.D))
true_xs = np.empty((args.K, args.T, args.D))
ys = np.empty((args.K, args.T, args.D))

for k, key_k in enumerate(tqdm.tqdm(EXPERIMENT_KEYS, desc="Experiment: ")):
    samples_k, s_ms_k, s_covs_k, true_xs_k, ys_k = one_experiment(key_k)
    
    # filter/smoother data
    samples[k] = samples_k
    means[k] = s_ms_k
    covs[k] = s_covs_k

    # true data
    true_xs[k] = true_xs_k
    ys[k] = ys_k

if not os.path.exists("results"):
    os.mkdir("results")

# save results
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
    os.mkdir(dirpath)

datapath = f"{dirpath}/data.npz"
np.savez_compressed(
    datapath, 
    smoothing_means=means,
    smoothing_covs=covs,
    samples=samples,
    true_xs=true_xs,
    ys=ys
)
