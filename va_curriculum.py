from imports import *
from gaussian_blur import *

class CurriculumScheduler:
    """
    Visual Acuity Curriculum from Paper (Algorithm 1)
    σ₀=16, k'=10, k=5
    """
    def __init__(self, sigma0=16, k_prime=10, k=5):
        self.sigma = sigma0
        self.k_prime = k_prime
        self.k = k

    def step(self, epoch):
        if epoch > self.k_prime and (epoch - self.k_prime) % self.k == 0:
            self.sigma = max(0, self.sigma // 2)
        return self.sigma

    def get_transform(self, training=True):
        if training:
            transform_list = []
            if self.sigma > 0:
                transform_list.append(GaussianBlur(self.sigma))

            transform_list.extend([
                transforms.Resize((512, 512)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.3),
                transforms.RandomRotation(10),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
        else:
            transform_list = [
                transforms.Resize((512, 512)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ]

        return transforms.Compose(transform_list)