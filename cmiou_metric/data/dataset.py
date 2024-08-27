import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class SegmentationDataset(Dataset):
    def __init__(self, images, masks, class_names, transform=None):
        self.images = images
        self.masks = masks
        self.class_names = class_names
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(self.images[idx]).convert("RGB")
        mask = Image.open(self.masks[idx])
        class_name = self.class_names[idx]

        if self.transform:
            image = self.transform(image)
            mask = transforms.ToTensor()(mask)

        return image, mask, class_name


def load_pascal_voc(root_dir):
    """
    加载PASCAL VOC数据集
    """
    image_dir = os.path.join(root_dir, "JPEGImages")
    mask_dir = os.path.join(root_dir, "SegmentationClass")

    images = sorted(
        [
            os.path.join(image_dir, f)
            for f in os.listdir(image_dir)
            if f.endswith(".jpg")
        ]
    )
    masks = sorted(
        [os.path.join(mask_dir, f) for f in os.listdir(mask_dir) if f.endswith(".png")]
    )

    with open(os.path.join(root_dir, "class_names.txt"), "r") as f:
        class_names = [line.strip() for line in f.readlines()]

    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    return SegmentationDataset(images, masks, class_names, transform)


def load_cityscapes(root_dir):
    """
    加载Cityscapes数据集
    """
    image_dir = os.path.join(root_dir, "leftImg8bit", "val")
    mask_dir = os.path.join(root_dir, "gtFine", "val")

    images = []
    masks = []
    class_names = []

    for city in os.listdir(image_dir):
        city_images = sorted(
            [
                os.path.join(image_dir, city, f)
                for f in os.listdir(os.path.join(image_dir, city))
                if f.endswith(".png")
            ]
        )
        city_masks = sorted(
            [
                os.path.join(
                    mask_dir, city, f.replace("leftImg8bit", "gtFine_labelIds")
                )
                for f in os.listdir(os.path.join(image_dir, city))
                if f.endswith(".png")
            ]
        )

        images.extend(city_images)
        masks.extend(city_masks)
        class_names.extend(
            ["cityscapes"] * len(city_images)
        )  # Cityscapes has a fixed set of classes

    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    return SegmentationDataset(images, masks, class_names, transform)


def load_ade20k(root_dir):
    """
    加载ADE20K数据集
    """
    image_dir = os.path.join(root_dir, "images", "validation")
    mask_dir = os.path.join(root_dir, "annotations", "validation")

    images = sorted(
        [
            os.path.join(image_dir, f)
            for f in os.listdir(image_dir)
            if f.endswith(".jpg")
        ]
    )
    masks = sorted(
        [
            os.path.join(mask_dir, f.replace(".jpg", ".png"))
            for f in os.listdir(image_dir)
            if f.endswith(".jpg")
        ]
    )

    with open(os.path.join(root_dir, "objectInfo150.txt"), "r") as f:
        class_names = [
            line.strip().split(",")[0] for line in f.readlines()[1:]
        ]  # Skip header

    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    return SegmentationDataset(images, masks, class_names, transform)
