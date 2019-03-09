from __future__ import absolute_import

from torchvision.transforms import *
from PIL import Image
import random
import numpy as np

class Random2DTranslation(object):
    """
    With a probability, first increase image size to (1 + 1/8), and then perform random crop.

    Args:
        height (int): target height.
        width (int): target width.
        p (float): probability of performing this transformation. Default: 0.5.
    """
    def __init__(self, height, width, p=0.5, interpolation=Image.BILINEAR):
        self.height = height
        self.width = width
        self.p = p
        self.interpolation = interpolation

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be cropped.

        Returns:
            PIL Image: Cropped image.
        """
        if random.random() < self.p: # directly resize as desired size
            return img.resize((self.width, self.height), self.interpolation)
        # First enlarge as 1.125x, then randomly cropped
        new_width, new_height = int(round(self.width * 1.125)), int(round(self.height * 1.125))
        resized_img = img.resize((new_width, new_height), self.interpolation)
        x_maxrange = new_width - self.width
        y_maxrange = new_height - self.height
        x1 = int(round(random.uniform(0, x_maxrange)))
        y1 = int(round(random.uniform(0, y_maxrange)))
        croped_img = resized_img.crop((x1, y1, x1 + self.width, y1 + self.height))
        return croped_img

if __name__ == '__main__':
    img = Image.open('/data2/market1501/bounding_box_train/0002_c1s1_000451_03.jpg')
    transform = Random2DTranslation(height=256, width=128, p=0.5)
    #img_t = transform(img)

    # use another transform: append RandomHorizontalFlip
    from torchvision import transforms
    transform2 = transforms.Compose(
        [
            Random2DTranslation(height=256, width=128, p=0.5),
            transforms.RandomHorizontalFlip(0.5),
        ]

    )
    img_t = transform2(img)

    import matplotlib.pyplot as plt

    plt.figure(12)
    plt.subplot(121)
    plt.imshow(img)
    plt.subplot(122)
    plt.imshow(img_t)
    plt.show()