import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
    
imagenet_mean_std = [[0.485, 0.456, 0.406],[0.229, 0.224, 0.225]]

class SimSiamTransform():
    def __init__(self, image_size, mean_std=imagenet_mean_std):
        image_size = 224 if image_size is None else image_size
        p_blur = 0.5 if image_size > 32 else 0

        self.transform = T.Compose([
            T.RandomResizedCrop(image_size, scale=(0.2, 1.0)),
            T.RandomHorizontalFlip(),
            T.RandomApply([T.ColorJitter(0.4,0.4,0.4,0.1)], p=0.8),
            T.RandomGrayscale(p=0.2),
            T.RandomApply([T.GaussianBlur(kernel_size=image_size//20*2+1, sigma=(0.1, 2.0))], p=p_blur),
            T.ToTensor(),
            T.Normalize(*mean_std)
        ])
    def __call__(self, x):
        x1 = self.transform(x)
        x2 = self.transform(x)
        return x1, x2 











