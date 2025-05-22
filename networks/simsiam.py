import torch
import torch.nn as nn
import torch.nn.functional as F 
from torchvision.models import resnet50


def D(p, z, version='simplified'): # cosine similarity
    if version == 'original':
        z = z.detach() # stop gradient
        p = F.normalize(p, dim=1) 
        z = F.normalize(z, dim=1) 
        return -(p*z).sum(dim=1).mean()

    elif version == 'simplified':
        return - F.cosine_similarity(p, z.detach(), dim=-1).mean()
    else:
        raise Exception


class projection_MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim=256, out_dim=128):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.layer3 = nn.Sequential(
            nn.Linear(hidden_dim, out_dim),
            nn.BatchNorm1d(out_dim)
        )
        self.num_layers = 3
    def set_layers(self, num_layers):
        self.num_layers = num_layers

    def forward(self, x):
        if self.num_layers == 3:
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
        elif self.num_layers == 2:
            x = self.layer1(x)
            x = self.layer3(x)
        else:
            raise Exception
        return x 

# 预测器
class prediction_MLP(nn.Module):
    def __init__(self, in_dim=128, hidden_dim=256, out_dim=128): # bottleneck structure（128->512->128）
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Linear(hidden_dim, out_dim)
        
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x 

class SimSiam(nn.Module):
    def __init__(self, backbone=resnet50(), backbone_knn=resnet50()):
        super().__init__()
        
        self.backbone = backbone
        # self.backbone_knn = backbone_knn
        self.projector = projection_MLP(backbone.output_dim)

        self.encoder = nn.Sequential( # f encoder
            self.backbone,
            self.projector
        )
        self.predictor = prediction_MLP()

    def forward(self, x1, x2):
        f, h = self.encoder, self.predictor
        z1, z2 = f(x1), f(x2)
        p1, p2 = h(z1), h(z2)
        L = D(p1, z2) / 2 + D(p2, z1) / 2
        return {'loss': L}


if __name__ == "__main__":
    model = SimSiam()
    x1 = torch.randn((2, 3, 224, 224))
    x2 = torch.randn_like(x1)

    model.forward(x1, x2).backward()
    print("forward backwork check")

    z1 = torch.randn((200, 2560))
    z2 = torch.randn_like(z1)
    import time
    tic = time.time()
    print(D(z1, z2, version='original'))
    toc = time.time()
    print(toc - tic)
    tic = time.time()
    print(D(z1, z2, version='simplified'))
    toc = time.time()
    print(toc - tic)













