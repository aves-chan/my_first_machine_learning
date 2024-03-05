import albumentations as albu
from albumentations.pytorch import ToTensorV2


def get_transforms(
    width: int,
    height: int,
    preprocessing: bool = True,
    augmentations: bool = True,
    postprocessing: bool = True,
) -> albu.BaseCompose:
    transforms = []

    if preprocessing:
        transforms.append(albu.Resize(height=height, width=width))

    if augmentations:
        loaded_transform = albu.load(
            "configs/augmentations_conf.yaml",
            data_format="yaml",
        )
        transforms_list = loaded_transform.transforms
        transforms.extend(
            transforms_list,
        )

    if postprocessing:
        transforms.extend([albu.Normalize(), ToTensorV2()])

    return albu.Compose(transforms)
