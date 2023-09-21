import albumentations as albu
from albumentations.pytorch import ToTensorV2
from torchvision.transforms import transforms


def inv_trans(tensor):
    inv_norm = transforms.Compose(
        [
            transforms.Normalize(mean=[0.0, 0.0, 0.0], std=[1 / 0.229, 1 / 0.224, 1 / 0.225]),
            transforms.Normalize(mean=[-0.485, -0.456, -0.406], std=[1.0, 1.0, 1.0]),
        ],
    )
    return inv_norm(tensor)


class Transforms:
    def __init__(self, img_height, img_width):
        self.img_height = img_height
        self.img_width = img_width

    def get_train_transforms(self):
        return [
            albu.Resize(height=self.img_height, width=self.img_width),
            albu.HorizontalFlip(),
            # albu.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),
            # albu.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            # albu.ShiftScaleRotate(),
            # albu.GaussianBlur(),
            albu.CenterCrop(height=self.img_height, width=self.img_width, p=0.5),
            # albu.ColorJitter(),
            # albu.RandomRotate90(),
            albu.VerticalFlip(),
            albu.Normalize(),
            ToTensorV2(),
        ]

    def get_val_transforms(self):
        return [
            albu.Resize(height=self.img_height, width=self.img_width),
            albu.Normalize(),
            ToTensorV2(),
        ]

    def compose(self, stage):
        if stage == 'fit':
            return albu.Compose(self.get_train_transforms())
        else:
            return albu.Compose(self.get_val_transforms())
