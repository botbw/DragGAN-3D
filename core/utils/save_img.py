import torch
from torchvision.utils import save_image


def save_eg3d_img(raw_img: torch.Tensor, path: str = 'my_image.jpg'):
    raw_img = raw_img.detach()
    raw_img = (raw_img + 1) / 2
    save_image(raw_img, path)


@torch.no_grad
def save_3d_depth_img(img_depth, path: str = 'depth_img.jpg'):
    import matplotlib.pyplot as plt
    import numpy
    from mpl_toolkits.mplot3d import Axes3D

    def gaussian_kernel(size: int, sigma: float):
        x = torch.arange(-size // 2 + 1., size // 2 + 1.).cuda()
        y = x.view(size, 1)
        x = x / sigma
        y = y / sigma

        return torch.exp(-(x**2 + y**2) / 2)

    def get_gauss_filter(in_channels: int, out_channels: int, kernel_size: int, sigma: float):
        kernel = gaussian_kernel(kernel_size, sigma)
        kernel = kernel / torch.sum(kernel)

        filter = torch.nn.Conv2d(in_channels,
                                 out_channels,
                                 kernel_size=kernel_size,
                                 stride=1,
                                 padding=kernel_size // 2,
                                 bias=False).cuda()
        filter.weight.data = kernel.view(1, 1, kernel_size,
                                         kernel_size).repeat(out_channels, in_channels, 1,
                                                             1).cuda()
        filter.weight.requires_grad = False

        return filter

    gauss_filter = get_gauss_filter(1, 1, 3, 1.0)

    # Set up grid and test data
    nx, ny = 128, 128
    x = range(nx)
    y = range(ny)

    img_depth = gauss_filter(img_depth[0])[0].detach().cpu().numpy()

    hf = plt.figure()
    ha = hf.add_subplot(111, projection='3d')

    # `plot_surface` expects `x` and `y` data to be 2D
    X, Y = numpy.meshgrid(x, y)
    ha.plot_surface(X, Y, img_depth)

    plt.savefig(path)
