from torchvision import transforms

# Stable normalization (good for attacks/defenses too)
CELEBA_MEAN = [0.5, 0.5, 0.5]
CELEBA_STD = [0.5, 0.5, 0.5]


def train_tfms(image_size: int = 128):
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=CELEBA_MEAN, std=CELEBA_STD),
        ]
    )


def eval_tfms(image_size: int = 128):
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=CELEBA_MEAN, std=CELEBA_STD),
        ]
    )
