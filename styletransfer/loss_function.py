import torch


def calc_2_moments(tensor: torch.Tensor):
    """flattens tensor and calculates sample mean and covariance matrix
    along last dim (presumably channels)

    Arguments:
        tensor {[torch.Tensor]} -- Tensor with shape(..., chan)
    """
    flatten_t = tensor.reshape(-1, tensor.size(-1))
    mu = torch.mean(flatten_t, dim=0, keepdim=True)
    flatten_t = flatten_t - mu
    cov = flatten_t.t().mm(flatten_t) / (flatten_t.size(0) - 1)
    return mu, cov


def calc_style_desc(activations: torch.Tensor, take_root: bool = False):
    """Get the style description tensors from a actvation tensor

    Arguments:
        activations {torch.Tensor} -- Activation feature map from VGG
        take_root   {bool}         -- Where to return the root of the covariance
    """
    mu, cov = calc_2_moments(activations)

    eigvals, eigvects = torch.symeig(cov, eigenvectors=True)
    tr_cov = torch.sum(eigvals)
    if take_root:
        eigroot_mat = torch.diag(torch.sqrt(torch.clamp(eigvals, 0)))
        root_cov = eigvects.mm(eigroot_mat).mm(eigvects.t())
        return mu, tr_cov, root_cov
    return mu, tr_cov, cov


def calc_l2_wass_dist(content_desc, style_desc):
    cont_mu, cont_tr_cov, cont_cov = content_desc
    style_mu, style_tr_cov, style_root_cov = style_desc

    cov_prod = style_root_cov.mm(cont_cov).mm(style_root_cov)
    # trace of sqrt of matrix is sum of sqrts of eigenvalues
    var_overlap = torch.sqrt(torch.clamp(torch.symeig(
        cov_prod, eigenvectors=True)[0], 1e-5)).sum()

    mu_diff_squared = torch.sum((cont_mu - style_mu) ** 2)

    dist = mu_diff_squared + cont_tr_cov + style_tr_cov - 2 * var_overlap
    return dist
