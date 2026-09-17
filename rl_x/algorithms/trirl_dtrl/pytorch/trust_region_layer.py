import torch


class KLCovProjection(torch.autograd.Function):
    @staticmethod
    def forward(ctx, pred_std, target_std, eps):
        eps = torch.as_tensor(eps, dtype=pred_std.dtype, device=pred_std.device)

        def covariance_kl(eta):
            covariance = (eta + 1) / torch.clamp(eta / target_std.square() + pred_std.reciprocal().square(), min=1e-8)
            std = torch.sqrt(torch.clamp(torch.nan_to_num(covariance), min=1e-16))
            return 0.5 * torch.sum(2 * (target_std.log() - std.log()) + (std / target_std).square() - 1)

        lower = torch.zeros((), dtype=pred_std.dtype, device=pred_std.device)
        upper = torch.ones((), dtype=pred_std.dtype, device=pred_std.device)
        for iteration in range(50):
            upper = torch.where(covariance_kl(upper) > eps, upper * 2, upper)
        for iteration in range(60):
            middle = 0.5 * (lower + upper)
            lower, upper = torch.where(covariance_kl(middle) > eps, middle, lower), torch.where(covariance_kl(middle) > eps, upper, middle)
        eta = upper
        projected_covariance = (eta + 1) / torch.clamp(eta / target_std.square() + pred_std.reciprocal().square(), min=1e-8)
        projected_covariance = torch.clamp(torch.nan_to_num(projected_covariance), min=1e-16)
        ctx.save_for_backward(pred_std, target_std, projected_covariance, eta)
        return projected_covariance, eta


    @staticmethod
    def backward(ctx, projected_covariance_gradient, eta_gradient):
        pred_std, target_std, projected_covariance, eta = ctx.saved_tensors
        d_q_d_eta = (target_std.reciprocal().square() - pred_std.reciprocal().square()) / (eta + 1)
        f2_d_q = projected_covariance * (1 - projected_covariance / target_std.square())
        sum_value = torch.sum(f2_d_q * d_q_d_eta)
        denominator = torch.where(torch.abs(sum_value) < 1e-8, torch.sign(sum_value) * 1e-8, sum_value + 1e-12)
        d_eta_d_q_pred = -f2_d_q / denominator
        d_q_d_eta = (target_std.reciprocal().square() - pred_std.reciprocal().square()) / (eta + 1).square()
        d_q = -projected_covariance * projected_covariance_gradient * projected_covariance
        d_eta = torch.sum(d_q * d_q_d_eta)
        d_q_pred = d_eta * d_eta_d_q_pred + d_q / (eta + 1)
        d_cov_pred = -pred_std.reciprocal().square() * d_q_pred * pred_std.reciprocal().square()
        d_cov_pred = torch.clamp(torch.nan_to_num(d_cov_pred), min=1e-20)
        return 2 * pred_std * d_cov_pred, None, None


def kl_projection(mean, std, old_mean, old_std, mean_bound, cov_bound):
    mean_part = 0.5 * torch.sum(((old_mean - mean) / old_std).square(), dim=-1)
    omega = torch.where(mean_part > mean_bound, torch.sqrt(torch.clamp(mean_part, min=mean_bound) / mean_bound) - 1, torch.zeros_like(mean_part))
    projected_mean = (mean + omega.unsqueeze(-1) * old_mean) / (1 + omega.unsqueeze(-1))
    projected_mean_part = (0.5 * torch.sum(((old_mean - projected_mean) / old_std).square(), dim=-1)).detach()

    cov_part = 0.5 * torch.sum(2 * (old_std.log() - std.log()) + (std / old_std).square() - 1)
    if cov_part.detach() > cov_bound:
        projected_covariance, eta_cov = KLCovProjection.apply(std.squeeze(0), old_std.squeeze(0), cov_bound)
        projected_std = projected_covariance.sqrt().unsqueeze(0)
    else:
        projected_std = std
        eta_cov = torch.zeros((), dtype=std.dtype, device=std.device)
    projected_cov_part = (0.5 * torch.sum(2 * (old_std.log() - projected_std.log()) + (projected_std / old_std).square() - 1)).detach()
    return projected_mean, projected_std, omega, eta_cov, mean_part, projected_mean_part, cov_part, projected_cov_part


def entropy_projection(action_logstd, beta, dim):
    entropy = torch.sum(action_logstd + 0.5 * torch.log(torch.as_tensor(2 * torch.pi * torch.e, dtype=action_logstd.dtype, device=action_logstd.device)))
    return torch.where(entropy < beta, action_logstd + (beta - entropy) / dim, action_logstd)
