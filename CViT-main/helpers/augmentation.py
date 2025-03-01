from albumentations import (
    HorizontalFlip, VerticalFlip, ShiftScaleRotate, CLAHE, RandomRotate90,
    Transpose, ShiftScaleRotate, HueSaturationValue,  
    GaussNoise, Sharpen, Emboss, RandomBrightnessContrast, OneOf, Compose, Blur, MotionBlur, MedianBlur
)
import numpy as np
from PIL import Image

def strong_aug(p=.5):
    return Compose([
        RandomRotate90(p=0.2),
        Transpose(p=0.2),
        HorizontalFlip(p=0.5),   
        VerticalFlip(p=0.5),
        OneOf([
            GaussNoise(),
        ], p=0.2),
        ShiftScaleRotate(p=0.2),
        OneOf([
            CLAHE(clip_limit=2),
            Sharpen(),
            Emboss(),
            RandomBrightnessContrast(),
        ], p=0.2),
        HueSaturationValue(p=0.2),
    ], p=p)

# def strong_aug(p=.5):
#     return Compose([
#         RandomRotate90(p=0.2),
#         HorizontalFlip(p=0.5),
#         VerticalFlip(p=0.5),
#         ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=30, p=0.5),
#         HueSaturationValue(p=0.3),
#         RandomBrightnessContrast(p=0.3),
#         GaussNoise(var_limit=(10.0, 50.0), p=0.2),
#         Blur(blur_limit=3, p=0.1),
#         MotionBlur(blur_limit=3, p=0.1),
#         MedianBlur(blur_limit=3, p=0.1),
#         CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.2),
#     ], p=p)


def augment(aug, image):
    return aug(image=image)['image']

class Aug(object):
    def __call__(self, img):
        aug = strong_aug(p=0.9)
        return Image.fromarray(augment(aug, np.array(img)))
