import torch
import torch.nn as nn



class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of
    residual blocks).

    Args:
        drop_prob (float): Drop rate for paths of model. Dropout rate has
            to be between 0 and 1. Default: 0.
    """

    def __init__(self, drop_prob=0.):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.keep_prob = 1 - drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        shape = (x.shape[0], ) + (1, ) * (
            x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
        random_tensor = self.keep_prob + torch.rand(
            shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # binarize
        output = x.div(self.keep_prob) * random_tensor
        return output


class Mlp(nn.Module): # FC层构成的前向网络
    def __init__(self, in_features, hidden_features,drop=0.):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x




class CMlp(nn.Module):  # 卷积层构成的前向网络
    def __init__(self, in_features, hidden_features,drop=0.):
        super().__init__()
        self.fc1 = nn.Conv1d(in_features, hidden_features, 1)
        self.act = nn.GELU()
        self.fc2 = nn.Conv1d(hidden_features, in_features, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class CBlock(nn.Module):  #
    def __init__(self, d_model, mlp_ratio=4., drop=0.,drop_path=0.):
        super().__init__()
        #使用深度卷积进行位置编码
        self.pos_embed = nn.Conv1d(d_model, d_model, 3, padding=1, groups=1)
        self.norm1 = nn.BatchNorm1d(d_model)
        self.conv1 = nn.Conv1d(d_model, d_model, 1)
        self.conv2 = nn.Conv1d(d_model, d_model, 1)


        # 使用CNN做self-attention
        self.attn = nn.Conv1d(d_model, d_model, 5, padding=2, groups=1)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = nn.BatchNorm1d(d_model)
        mlp_hidden_dim = int(d_model * mlp_ratio)
        self.mlp = CMlp(in_features=d_model, hidden_features=mlp_hidden_dim, drop=drop)

    def forward(self, x):
        x = x.transpose(1,2)
        x = x + self.pos_embed(x)
        x = x + self.drop_path(self.conv2(self.attn(self.conv1(self.norm1(x)))))
        # x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x.transpose(1,2)

if __name__ == '__main__':
    x = torch.rand((32,512,512))
    CBlockimp = CBlock(512)
    CBlockimp(x)
    print(x.shape)