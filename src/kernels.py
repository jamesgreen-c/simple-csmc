from enum import Enum
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.stats import norm

import utils.csmc as csmc
from src.model import log_likelihood, log_potential, log_pdf


class KernelType(Enum):
    CSMC = 0

    @property
    def kernel_maker(self):
        if self == KernelType.CSMC:
            return get_csmc_kernel
        else:
            raise NotImplementedError

    def shape_delta(self, delta, T):
        if self == KernelType.CSMC:
            return delta
        else:
            raise NotImplementedError


#######################
# Kernel constructors #
#######################

def get_csmc_kernel(ys, sigma, N, style="bootstrap", **kwargs):
    T, dx = ys.shape

    if style == "bootstrap":
        def M0_rvs(key, _):
            eps = jax.random.normal(key, (N + 1, dx))
            return sigma * eps

        def Mt_rvs(key, x_t_m_1, _):
            eps = jax.random.normal(key, (N + 1, dx))
            return x_t_m_1 + sigma * eps

        M0_logpdf = lambda x: norm.logpdf(x, scale=sigma).sum()
        M0_logpdf = jnp.vectorize(M0_logpdf, signature="(d)->()")
        Mt_logpdf = lambda x_t_m_1, x_t, _params: norm.logpdf(x_t, x_t_m_1, sigma).sum()
        Mt_logpdf = jnp.vectorize(Mt_logpdf, signature="(d),(d)->()", excluded=(2,))
        Gamma_0 = lambda x: log_potential(x, ys[0]) + M0_logpdf(x)
        Gamma_t = lambda x_t_m_1, x_t, y: log_potential(x_t, y) + Mt_logpdf(x_t_m_1, x_t, None)

    else:
        raise NotImplementedError(f"Unknown style: {style}, choose from 'bootstrap'")

    M0 = M0_rvs, M0_logpdf
    Mt = Mt_rvs, Mt_logpdf, ys[1:]
    Gamma_t_plus_params = Gamma_t, ys[1:]

    kernel = lambda key, state, *_: csmc.kernel(key, state[0], state[1], M0, Gamma_0, Mt, Gamma_t_plus_params, N=N,
                                                **kwargs)
    init = lambda x: (x, jnp.zeros((x.shape[0],), dtype=int))

    return kernel, init
