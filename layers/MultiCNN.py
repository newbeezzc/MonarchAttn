import torch
import torch.nn as nn

from layers.CBlock import CBlock


class MultiCNN(nn.Module):
    def __init__(self, kernel_size=96, d_model=512, n_heads=8, depth=1):
        super(MultiCNN, self).__init__()
        self.blocks = nn.ModuleList([
            CBlock(d_model=d_model) for i in range(depth)
        ])

    def forward(self, x, x1, x2, attn_mask=None, tau=None, delta=None):
        # 32 96 512
        for cblock in self.blocks:
            x = cblock(x)

        return x, None


if __name__ == '__main__':
    x = torch.rand(32, 96, 512)
    print(x.shape)
    Dy = MultiCNN()
    x, _ = Dy(x, x, x)
    print(x.shape)
