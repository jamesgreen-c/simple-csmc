from typing import Any

import jax
import jax.numpy as jnp
from chex import Array, PRNGKey

from jax.scipy.special import logsumexp


def ess(ws: Array, log_weights: bool = True):
    """
    Calculate the ESS = 1 / sum(ws**2)
    If log_weights: calculate from logspace
    
    :param ws: weights (T, N)
    :param log_weights: Calculate ESS from logspace?
    """
    if log_weights:
        a = logsumexp(ws, axis=1)          # (T,)
        b = logsumexp(2.0 * ws, axis=1)    # (T,)
        return jnp.exp(2.0 * a - b)        # (T,)

    ws = ws / jnp.sum(ws, axis=1, keepdims=True)  # (T, N) normalise per t
    return 1.0 / jnp.sum(ws**2, axis=1)           # (T,)


def force_move(key: PRNGKey, weights: Array, k: int) -> [Array, float]:
    """
    Forced-move trajectory selection. The weights are assumed to be normalised already.

    Parameters
    ----------
    key:
        Random number generator key.
    weights:
        Log-weights of the particles.
    k:
        Index of the reference particle.

    Returns
    -------
    l_T:
        New index of the ancestor of the reference particle.
    alpha:
        Probability of accepting new sample.
    """
    # TODO: log space?
    M = weights.shape[0]
    key_1, key_2 = jax.random.split(key, 2)

    w_k = weights[k]
    temp = 1 - w_k

    rest_weights = weights.at[k].set(0)  # w_{-k}
    threshold = jnp.maximum(1 - jnp.exp(-M), 1 - 1e-12)
    rest_weights = jax.lax.cond(w_k < threshold, lambda: rest_weights / temp,
                                lambda: jnp.full((M,), 1 / M))  # w_{-k} / (1 - w_k)

    i = jax.random.choice(key_1, M, p=rest_weights, shape=())  # i ~ Cat(w_{-k} / (1 - w_k))
    u = jax.random.uniform(key_2, shape=())
    accept = u * (1 - weights[i]) < temp  # u < (1 - w_k) / (1 - w_i)

    alpha = jnp.nansum(temp * rest_weights / (1 - weights))
    i = jax.lax.select(accept, i, k)

    return i, jnp.clip(alpha, 0, 1.)


def barker_move(key: PRNGKey, weights: Array, k: Any = 0) -> [Array, float]:
    """
    Forced-move trajectory selection. The weights are assumed to be normalised already.

    Parameters
    ----------
    key:
        Random number generator key.
    weights:
        Log-weights of the particles.
    k:
        Index of the reference particle. Not used for the Barker move.

    Returns
    -------
    l_T:
        New index of the ancestor of the reference particle.
    alpha:
        Probability of accepting new sample.
    """
    M = weights.shape[0]
    i = jax.random.choice(key, M, p=weights, shape=())
    return i, 1 - weights[k]