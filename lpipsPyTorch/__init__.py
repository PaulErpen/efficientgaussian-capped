import torch
import lpips

lpips_alex = None
lpips_vgg = None
lpips_squeeze = None

def lpips(x: torch.Tensor,
          y: torch.Tensor,
          net_type: str = 'alex'):
    r"""Function that measures
    Learned Perceptual Image Patch Similarity (LPIPS).

    Arguments:
        x, y (torch.Tensor): the input tensors to compare.
        net_type (str): the network type to compare the features: 
                        'alex' | 'squeeze' | 'vgg'. Default: 'alex'.
    """
    device = x.device
    if net_type == 'alex':
        global lpips_alex
        if lpips_alex is None:
            lpips_alex = lpips.LPIPS(net='alex').to(device)
        return lpips_alex(x, y)
    elif net_type == 'vgg':
        global lpips_vgg
        if lpips_vgg is None:
            lpips_vgg = lpips.LPIPS(net='vgg').to(device)
        return lpips_vgg(x, y)
    elif net_type == 'squeeze':
        global lpips_squeeze
        if lpips_squeeze is None:
            lpips_squeeze = lpips.LPIPS(net='squeeze').to(device)
        return lpips_squeeze(x, y)
    else:
        raise ValueError("Invalid net_type: {}".format(net_type))
