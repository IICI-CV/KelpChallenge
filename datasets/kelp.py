import os
import torch
import cv2
from torch.utils.data import Dataset
from torchvision import transforms as T
from PIL import Image
import numpy as np
import albumentations as A
import tifffile
import tqdm
# np.set_printoptions(suppress=True)  # 取消科学计数法
# torch.set_printoptions(precision=12, sci_mode=False)


def get_train_transform(img_size=512):
    trfm = A.Compose(
        [
            # A.Resize(img_size, img_size),
            A.RandomCrop(img_size, img_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.75),
            A.ShiftScaleRotate(p=0.7),
            A.Transpose(p=0.7),
        ]
    )
    return trfm

color_aug = A.OneOf(
    [
        A.RandomBrightnessContrast(p=0.6),
        A.RandomGamma(p=0.6),
        # A.RandomBrightness(p=0.6),
        A.ColorJitter(
            brightness=0.67,
            contrast=0.67,
            saturation=0.61,
            hue=0.21,
            always_apply=False,
            p=0.8,
        ),
    ],
    p=0.8,
)

class KelpDataset(Dataset):
    def __init__(
        self,
        dataset_root,
        ids_filepath,
        transform=get_train_transform(),
        test_mode=False,
        size=256,
    ):
        self.paths = None
        self.transform = get_train_transform(img_size=size)
        self.test_mode = test_mode
        self.data_root = dataset_root
        self.size = size

        with open(ids_filepath, "r") as f:
            self.paths = [x.strip() for x in f.readlines()]

        self.to_tensor = T.ToTensor()
        self.as_tensor = T.Compose(
            [
                T.ToTensor(),
                T.Normalize(
                    [
                        8310.221679687500,
                        8990.597656250000,
                        7060.784179687500,
                        7177.506347656250,
                        6858.071777343750,
                        0.0,
                        -1301.268066406250,
                    ],
                    [
                        9165.939453125000,
                        9644.202148437500,
                        8482.499023437500,
                        8487.715820312500,
                        8450.488281250000,
                        1.0,
                        6435.906250000000,
                    ],
                )]
        )

    # get data operation
    def __getitem__(self, index):
        # img = Image.open(self.data_root + "/" + self.paths[index])
        img = tifffile.imread(self.data_root + "/" + self.paths[index])
        # mask 读取为灰度图
        pathhh = self.paths[index].replace("_satellite", "_kelp")
        if os.path.exists(self.data_root + "/" + pathhh):
            mask = Image.open(self.data_root + "/" + pathhh)
        else:
            mask = Image.fromarray(np.zeros(img.shape[:2], dtype=np.uint8))

        # convert PIL.Image
        mask = mask.convert("L")
        # to numpy
        img = np.array(img, dtype=np.float32)
        mask = np.array(mask, dtype=np.float32)
        mask[mask > 0.5] = 1
        mask[mask <= 0.5] = 0

        if not self.test_mode:
            
            # need_aug_img = img[:, :, :5]
            
            augments = (
                self.transform(image=img, mask=mask)
                if self.transform
                else {"image": img, "mask": mask}
            )
            img, mask = augments["image"], augments["mask"]
            for i_aug_c in range(5):
                img_temp = img[:, :, i_aug_c]
                # img_temp[img_temp < 0] = 1e-4
                img_temp_min = img_temp.min()
                img[:, :, i_aug_c] = color_aug(image=img_temp - img_temp_min)["image"] + img_temp_min
            # 输出augments["mask"]的类型
            return (self.as_tensor(augments["image"]), self.to_tensor(augments["mask"]))
        else:
            # augments = self.crop(image=img, mask=mask)
            # img = augments["image"]
            # mask = augments["mask"]
            # print(mask[None].shape)
            return self.as_tensor(img).float(), self.to_tensor(mask).float()

    def __len__(self):
        """
        Total number of samples in the dataset
        """
        return len(self.paths)


def get_mean_std(loader):
    # Var[x] = E[X**2]-E[X]**2
    channels_sum, channels_squared_sum, num_batches = 0, 0, 0
    for data, _ in tqdm.tqdm(loader):
        channels_sum += torch.mean(data, dim=[0, 2, 3])
        channels_squared_sum += torch.mean(data ** 2, dim=[0, 2, 3])
        num_batches += 1

    # print(num_batches)
    # print(channels_sum)
    mean = channels_sum / num_batches
    std = (channels_squared_sum / num_batches - mean ** 2) ** 0.5

    return mean, std


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    import cv2

    wdataset = KelpDataset(
        dataset_root="/home/zb/WCS/codes/Kelp/datasets/datasets",
        ids_filepath="/home/zb/WCS/codes/Kelp/datasets/datasets/all_ids.txt",
        test_mode=False,
    )
    print(len(wdataset), wdataset[0][0].shape, wdataset[0][0].max())
    print("Final: ", get_mean_std(DataLoader(wdataset, drop_last=False, batch_size=1)))
