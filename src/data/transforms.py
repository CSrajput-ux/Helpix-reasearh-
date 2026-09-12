from torchvision import transforms


def get_training_transforms(img_size: int = 224) -> transforms.Compose:
    """
    Standard data augmentation for HELPix-R training:
    - Resize to 224x224
    - Random Horizontal and Vertical Flips
    - Random Rotation up to 20 degrees
    - ColorJitter (brightness=0.2, contrast=0.2, saturation=0.1)
    - ImageNet channel normalization
    """
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def get_evaluation_transforms(img_size: int = 224) -> transforms.Compose:
    """
    Deterministic preprocessing for evaluation and inference:
    - Resize to 224x224
    - ImageNet channel normalization
    """
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
