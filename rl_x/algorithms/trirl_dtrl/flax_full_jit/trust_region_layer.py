import jax
import jax.numpy as jnp
import numpy as np
import optax
from typing import Tuple
from jax.lax import stop_gradient
from jaxopt import LBFGS
from functools import partial

####################################################################
####################################################################
"""
Chunked Trust Region Projection 
"""
####################################################################
####################################################################

def make_chunked_compute_etas(policy_apply_fn, kl_cov_proj, nr_steps: int, nr_envs: int, mean_bound: float, cov_bound: float):
    """
    chunked_compute_etas(policy_buffer, inputs, chunk_size)
    states: (batch, ...)
    returns: etas (nr_pairs,) where nr_pairs = len(policy_buffer) - 1
    """

    vmapped_policy = jax.vmap(
        policy_apply_fn,
        in_axes=(None, 0),   # (states, params_batch)
        out_axes=(0, 0),
    )

    def kl_mean_proj(mean, old_mean, old_std, eps):
        maha = 0.5 * jnp.sum(((old_mean - mean) / old_std) ** 2)

        def proj(_):
            omega = jnp.sqrt(maha / eps) - 1.0
            proj_mean = (mean + omega * old_mean) / (1.0 + omega)
            post = jax.lax.stop_gradient(0.5 * jnp.sum(((old_mean - proj_mean) / old_std) ** 2))
            return proj_mean, omega, post

        def nop(_):
            return mean, 0.0, maha

        return jax.lax.cond(maha > eps, proj, nop, operand=None)

    vmapped_kl_mean = jax.vmap(kl_mean_proj, in_axes=(0, 0, 0, None), out_axes=(0, 0, 0))
    vmap_cov = jax.vmap(kl_cov_proj, in_axes=(0, 0, None, None))  # returns (proj, omega) or omega

    @partial(jax.jit, static_argnames=('chunk_size'))
    def process_chunk_on_device(
        etas, params_chunk, states,
        available: jnp.ndarray, chunk_size: int, write_index: jnp.ndarray
    ):

        mean, logstd = vmapped_policy(states, params_chunk)  # mean: (chunk_size, batch, A), logstd: (chunk_size, A)
        std = jnp.exp(logstd)  # (chunk_size, A)
        max_pairs = chunk_size - 1
        pair_count_valid = jnp.maximum(jnp.minimum(available, chunk_size) - 1, 0)

        def _no_pairs(_):
            return etas

        def _has_pairs(args):
            etas, mean, std, write_index, pair_count_valid = args

            old = mean[:-1] # (max_pairs, B, A)
            curr = mean[1:] # (max_pairs, B, A)
            old_std = std[:-1] # (max_pairs, A)
            curr_std = std[1:] # (max_pairs, A)
            batch = curr.shape[1]
            action_dim = curr.shape[2]

            # Mean constraint
            curr_flat = jnp.reshape(curr, (max_pairs * batch, action_dim))
            old_flat = jnp.reshape(old, (max_pairs * batch, action_dim))
            old_std_flat = jnp.repeat(old_std, repeats=batch, axis=0)

            _, omega_mu_flat, _ = vmapped_kl_mean(curr_flat, old_flat, old_std_flat, mean_bound)
            omega_mu = jnp.reshape(omega_mu_flat, (max_pairs, batch))  # (max_pairs, B)
            eta_mu = jax.lax.stop_gradient(omega_mu)

            # Covariance constraint
            cov_out = vmap_cov(curr_std, old_std, cov_bound, action_dim)
            omega_cov = cov_out[1] if isinstance(cov_out, (tuple, list)) else cov_out  # (max_pairs,)
            eta_cov = jax.lax.stop_gradient(omega_cov)

            # Combine
            eta_pair = jnp.reshape(
                jnp.maximum(eta_mu, eta_cov[:, None]), (max_pairs, nr_steps, nr_envs)
            )

            mask = (jnp.arange(max_pairs) < pair_count_valid)[:, None, None]
            eta_pair_masked = eta_pair * mask
            return jax.lax.dynamic_update_slice(etas, eta_pair_masked, (write_index, 0, 0))
    
        return jax.lax.cond((mean.shape[0] - 1) > 0, _has_pairs, _no_pairs,
                            operand=(etas, mean, std, write_index, pair_count_valid))

    @partial(jax.jit, static_argnames=('chunk_size', 'buffer_size'))
    def chunked_compute_etas(policy_buffer, states, level: jnp.ndarray, chunk_size: int, buffer_size: int):
        """
        policy_buffer: pytree with leading axis >= level (full TrainStateBuffer)
        states: states
        level:  dynamic int32, number of valid params to use (B)
        """
        nr_policies = jnp.asarray(level, dtype=jnp.int32)
        nr_pairs = jnp.maximum(nr_policies - 1, 0)
        stride = chunk_size - 1
        n_chunks = jnp.where(
            nr_policies <= 1,
            1,
            ((nr_policies - 1) + stride - 1) // stride
        )

        def _no_pairs(_):
            return jnp.zeros((buffer_size, nr_steps, nr_envs), dtype=jnp.float32)

        def _pairs(_):
            etas = jnp.zeros((buffer_size, nr_steps, nr_envs), dtype=jnp.float32)

            def loop_body(k, etas):
                start = k * stride
                available = jnp.minimum(chunk_size, nr_policies - start).astype(jnp.int32)

                params_chunk = jax.tree.map(
                    lambda x: jax.lax.dynamic_slice_in_dim(x, start_index=start, slice_size=chunk_size),
                    policy_buffer
                )

                etas = process_chunk_on_device(
                    etas, params_chunk, states,
                    available=available, chunk_size=chunk_size, write_index=start
                )
                return etas

            etas = jax.lax.fori_loop(0, n_chunks, loop_body, etas)
            return jax.lax.stop_gradient(etas)

        etas = jax.lax.cond(nr_pairs > 0, _pairs, _no_pairs, operand=None)

        mask = (jnp.arange(buffer_size, dtype=level.dtype) < level)
        etas = etas * mask.reshape(buffer_size, *([1] * (etas.ndim - 1))).astype(etas.dtype)
        return etas

    return chunked_compute_etas


####################################################################
####################################################################
"""
KL divergence projection layer from https://arxiv.org/pdf/2101.09207
"""
####################################################################
####################################################################


def dual(eta_omega, pred_std, target_std, target_logdet, eps, omega_offset):

    eta = jnp.where(eta_omega[0] > 0.0, eta_omega[0], 0.0)
    new_std = jnp.sqrt((eta + omega_offset) / jnp.clip((eta/(target_std**2)) + (1/pred_std)**2, min=1e-8))
    new_std = jnp.clip(jnp.nan_to_num(new_std), min=1e-8)
    new_logdet = -2.0 * jnp.sum(jnp.log(1/new_std))

    dual_val = eta * eps - 0.5 * eta * target_logdet
    dual_val += 0.5 * (eta + omega_offset) * new_logdet

    kl = 0.5 * jnp.sum(2.0 * (jnp.log(target_std) - jnp.log(new_std)) + (new_std/target_std)**2 - 1.0)
    grad_val = eps - kl

    return dual_val, jnp.array([grad_val])

@jax.custom_vjp
def kl_cov_proj(pred_std: jnp.ndarray, target_std: jnp.ndarray, eps: float, max_eval: int = 50, 
                        omega_offset: float = 1.0, eta_init: float = 0.0) -> Tuple[jnp.ndarray, float]:

    """
    pred_std: standard deviation of the policy's prediction:  (1, as_dim)
    target_std: standard deviation of the old_policy:  (1, as_dim)
    max_eval: number of iterations of L-BFGS
    omega_offset: offset to include entropy term in the original optimization objective (see pg: 17 in paper)
    eta_init: initial guess for L-BFGS
    """
    target_logdet = -2.0 * jnp.sum(jnp.log(jnp.clip(1/target_std, min=1e-8)))

    """ Optax L BFGS """
    def objective(eta_omega):
        val, grad = dual(eta_omega, pred_std, target_std, target_logdet, eps, omega_offset)
        return val, grad

    def opt_bfgs(init_params, fun, opt, max_iter, tol):

        value_and_grad_fun = objective

        def step(carry):
            params, state = carry
            value, grad = objective(params)
            updates, state = opt.update(
                grad, state, params, value=value, grad=grad, value_fn=fun
            )
            params = optax.apply_updates(params, updates)
            return params, state

        def continuing_criterion(carry):
            _, state = carry
            iter_num = optax.tree.get(state, 'count')
            grad = optax.tree.get(state, 'grad')
            err = optax.tree.norm(grad)
            return (iter_num == 0) | ((iter_num < max_iter) & (err >= tol))

        init_carry = (init_params, opt.init(init_params))
        final_params, final_state = jax.lax.while_loop(
            continuing_criterion, step, init_carry
        )
        return final_params, final_state


    # Initialize parameters
    init_params = jnp.array([eta_init])

    # Optax LBFGS optimizer setup
    lbfgs = optax.lbfgs()
    params, _ = opt_bfgs(init_params, lambda x: objective(x)[0], lbfgs, max_iter=max_eval, tol=1e-9)
    eta_opt = params[0]

    """ Computing projected cov """
    projected_cov = (eta_opt + omega_offset) / jnp.clip(eta_opt/(target_std**2) + (1.0/pred_std)**2, min=1e-8)
    projected_cov = jnp.clip(jnp.nan_to_num(projected_cov), min=1e-16)
    return projected_cov, eta_opt



def kl_cov_proj_backward(d_proj: jnp.ndarray,
                                succ: bool,
                                omega_offset: float,
                                eta: float,
                                pred_std: jnp.ndarray,
                                target_std: jnp.ndarray,
                                projected_std: jnp.ndarray) -> jnp.ndarray:

    if not succ:
        raise RuntimeError("Optimization was not successful, cannot run backward")

    def last_eta_grad(pred_std, target_std, projected_std, eta, omega_offset):

        def eta_zero_case(_):
            return jnp.zeros_like(target_std)

        def eta_positive_case(eta):
            dQ_deta = (omega_offset * (1/target_std**2) - (1/pred_std**2)) / (eta + omega_offset)
            tmp = jnp.ones_like(target_std) - (1/target_std**2) * projected_std**2
            f2_dQ = projected_std**2 * tmp
            sum_val = jnp.sum(f2_dQ * dQ_deta)
            c = -1.0 / jnp.where(jnp.abs(sum_val) < 1e-8, jnp.sign(sum_val) * 1e-8, sum_val + 1e-12)
            return c * f2_dQ

        def eta_negative_case(_):
            return jnp.full_like(target_std, jnp.nan)

        return jax.lax.cond(
            eta == 0.0,
            eta_zero_case,
            lambda _: jax.lax.cond(
                eta > 0.0,
                eta_positive_case,
                eta_negative_case,
                operand=eta
            ),
            operand=eta
        )

    deta_dQ_pred = last_eta_grad(pred_std, target_std, projected_std, eta, omega_offset)
    eo = omega_offset + eta
    eo_squared = eo * eo
    dQ_deta = (omega_offset * (1/target_std**2) - (1/pred_std**2)) / eo_squared
    d_Q = - (projected_std**2) * d_proj * (projected_std**2)
    d_eta = jnp.sum(d_Q * dQ_deta)
    d_Q_pred = d_eta * deta_dQ_pred + d_Q / eo
    d_cov_pred = - (1/pred_std**2) * d_Q_pred * (1/pred_std**2)  
    d_cov_pred = jnp.clip(jnp.nan_to_num(d_cov_pred), min=1e-20)
    d_pred_std = 2.0 * pred_std * d_cov_pred

    return d_pred_std

def kl_cov_proj_fwd(pred_std, target_std, eps, max_eval, omega_offset, eta_init):
    projected_cov, eta = kl_cov_proj(pred_std, target_std, eps, max_eval, omega_offset, eta_init)
    residuals = (pred_std, target_std, projected_cov, eta, omega_offset)
    return (projected_cov, eta), residuals

def kl_cov_proj_bwd(residuals, cotangents):
    d_projected_cov, d_eta = cotangents
    pred_std, target_std, projected_cov, eta, omega_offset = residuals
    projected_std = jnp.sqrt(projected_cov)
    d_std_pred = kl_cov_proj_backward(d_proj=d_projected_cov, succ=True, omega_offset=omega_offset, eta=eta, pred_std=pred_std, target_std=target_std, projected_std=projected_std)
    return (d_std_pred, None, None, None, None, None)

kl_cov_proj.defvjp(kl_cov_proj_fwd, kl_cov_proj_bwd)

def kl_projection(mean, std, mean_other, std_other, eps_mean, eps_cov):
    """
    Project the pred policy back to satisfy reverse KL --> KL(pi_old || pi)  = (old_mean - mean)* inv_cov * (old_mean - mean) + cov_kl

    Args:
        mean, std: mean and std of the current policy
        mean_other, std_other: mean and std of the old policy
    """

    mean_part = 0.5 * jnp.sum(((mean_other - mean)/ std_other) ** 2)

    def mean_projection(mean, old_mean, maha, std_other, eps):
        """
        Projects the mean based on the Mahalanobis objective and trust region.
        Args:
            mean: current mean vectors
            old_mean: old mean vectors
            mean_part: Mahalanobis distance between the two mean vectors
            eps_mean: trust region bound

        Returns:
            projected mean that satisfies the trust region
            lagrangian multipliers
        """

        def true_fn(_):
            omega = jnp.sqrt(maha / eps) - 1.0
            proj_mean = (mean + omega * old_mean) / (1 + omega)
            # kl contribution of the mean after projection
            post_proj_kl_mean_part = stop_gradient(0.5 * jnp.sum(((mean_other - proj_mean)/ std_other) ** 2))
            return proj_mean, omega, post_proj_kl_mean_part

        def false_fn(_):
            # Skip if already in the trust region
            return mean, 0.0, maha

        proj_mean, omega, post_proj_kl_mean_part = jax.lax.cond(maha > eps, true_fn, false_fn, operand=None)
        return proj_mean, omega, post_proj_kl_mean_part


    proj_mean, eta_mu, post_proj_kl_mean_part = mean_projection(mean, mean_other, mean_part, std_other, eps_mean)

    # Skip if already in the trust region
    kl_cov_part = 0.5 * jnp.sum(2.0 * (jnp.log(std_other) - jnp.log(std)) + (std/std_other)**2 - 1.0)
    def do_cov_proj(_):
        proj_cov, eta_cov = kl_cov_proj(jnp.squeeze(std), jnp.squeeze(std_other), eps_cov)
        proj_std = jnp.expand_dims(jnp.sqrt(proj_cov), 0)
        post_proj_kl_cov_part = stop_gradient(0.5 * jnp.sum(2.0 * (jnp.log(std_other) - jnp.log(proj_std)) + (proj_std/std_other)**2 - 1.0))
        return proj_std, eta_cov, post_proj_kl_cov_part

    def skip_cov_proj(_):
        return std, 0.0, kl_cov_part

    proj_std, eta_cov, post_proj_kl_cov_part = jax.lax.cond(
        kl_cov_part > eps_cov,
        do_cov_proj,
        skip_cov_proj,
        operand=None
    )

    return proj_mean, proj_std, eta_mu, eta_cov, mean_part, post_proj_kl_mean_part, kl_cov_part, post_proj_kl_cov_part


def entropy_projection(action_logstd, beta, dim):
    """
    Projects std to satisfy an entropy inequality constraint.
    """
    entropy = jnp.sum(action_logstd + 0.5 * jnp.log(2.0 * jnp.pi * jnp.e))

    def true_fn(_):
        logalpha = (beta - entropy)/dim
        proj_logstd = logalpha + action_logstd
        return proj_logstd

    def false_fn(_):
        return action_logstd

    proj_logstd = jax.lax.cond(entropy < beta, true_fn, false_fn, operand=None)
    return proj_logstd
