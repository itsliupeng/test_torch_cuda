import torch


def shift(x, n_segment, fold_div=3):
    nt, c, h, w = x.size()
    n_batch = int(nt / n_segment)
    x = x.view(n_batch, n_segment, c, h, w)
    fold = int(c / fold_div)
    left_side = torch.cat((x[:, 1:, :fold], torch.zeros(n_batch, 1, fold, h, w).to(x.device)), dim=1)
    middle_side = torch.cat((torch.zeros(n_batch, 1, fold, h, w).to(x.device), x[:, :n_segment - 1, fold: 2 * fold]),
                            dim=1)
    out = torch.cat((left_side, middle_side, x[:, :, 2 * fold:]), dim=2)
    return out.view(nt, c, h, w)


if __name__ == '__main__':
    nt = 2
    c = 3
    h = 2
    w = 2
    n_segment = nt
    fold_div = c
    x = torch.arange(nt*c*h*w).float().reshape([nt, c, h, w]);
    out = shift(x, n_segment, fold_div)
    print(x)
    print(out)
