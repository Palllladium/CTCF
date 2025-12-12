import torch
from torch import nn


def get_mc_preds(net, inputs, mc_iter: int = 25):
    img_list, flow_list = [], []
    with torch.no_grad():
        for _ in range(mc_iter):
            img, flow = net(inputs)
            img_list.append(img)
            flow_list.append(flow)
    return img_list, flow_list


def calc_error(tar, img_list):
    sqr_diffs = [(img - tar) ** 2 for img in img_list]
    return torch.mean(torch.cat(sqr_diffs, dim=0), dim=0, keepdim=True)


def get_mc_preds_w_errors(net, inputs, target, mc_iter: int = 25):
    img_list, flow_list, err = [], [], []
    mse = nn.MSELoss()
    with torch.no_grad():
        for _ in range(mc_iter):
            img, flow = net(inputs)
            img_list.append(img)
            flow_list.append(flow)
            err.append(mse(img, target).item())
    return img_list, flow_list, err


def get_diff_mc_preds(net, inputs, mc_iter: int = 25):
    img_list, flow_list, disp_list = [], [], []
    with torch.no_grad():
        for _ in range(mc_iter):
            img, _, flow, disp = net(inputs)
            img_list.append(img)
            flow_list.append(flow)
            disp_list.append(disp)
    return img_list, flow_list, disp_list


def uncert_regression_gal(img_list, reduction="mean"):
    img_list = torch.cat(img_list, dim=0)
    ale = img_list[:, -1:].mean(dim=0, keepdim=True)
    epi = torch.var(img_list[:, :-1], dim=0, keepdim=True).mean(dim=1, keepdim=True)
    uncert = ale + epi
    if reduction == "mean":
        return ale.mean().item(), epi.mean().item(), uncert.mean().item()
    if reduction == "sum":
        return ale.sum().item(), epi.sum().item(), uncert.sum().item()
    return ale.detach(), epi.detach(), uncert.detach()


def uceloss(errors, uncert, n_bins=15, outlier=0.0, range=None):
    device = errors.device
    if range is None:
        bin_boundaries = torch.linspace(uncert.min().item(), uncert.max().item(), n_bins + 1, device=device)
    else:
        bin_boundaries = torch.linspace(range[0], range[1], n_bins + 1, device=device)

    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    uce = torch.zeros(1, device=device)
    errors_in_bin_list, avg_uncert_in_bin_list, prop_in_bin_list = [], [], []

    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = uncert.gt(bin_lower.item()) * uncert.le(bin_upper.item())
        prop_in_bin = in_bin.float().mean()
        prop_in_bin_list.append(prop_in_bin)
        if prop_in_bin.item() > outlier:
            errors_in_bin = errors[in_bin].float().mean()
            avg_uncert_in_bin = uncert[in_bin].mean()
            uce += torch.abs(avg_uncert_in_bin - errors_in_bin) * prop_in_bin
            errors_in_bin_list.append(errors_in_bin)
            avg_uncert_in_bin_list.append(avg_uncert_in_bin)

    err_in_bin = torch.tensor(errors_in_bin_list, device=device)
    avg_uncert_in_bin = torch.tensor(avg_uncert_in_bin_list, device=device)
    prop_in_bin = torch.tensor(prop_in_bin_list, device=device)
    return uce, err_in_bin, avg_uncert_in_bin, prop_in_bin