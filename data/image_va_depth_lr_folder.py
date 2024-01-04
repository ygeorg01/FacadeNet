"""A modified image folder class

We modify the official PyTorch image folder (https://github.com/pytorch/vision/blob/master/torchvision/datasets/folder.py)
so that this class can load images from both current directory and its subdirectories.
"""

import torch.utils.data as data
import torch

from PIL import Image
import os
import os.path

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
    '.tif', '.TIF', '.tiff', '.TIFF', '.webp',
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def make_dataset(dir, dir_h, dir_v, dir_sem, dir_depth, dir_feats, fnames_list, max_dataset_size=float("inf")):

    images = []

    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in os.walk(dir, followlinks=True):
        for fname in fnames:

            if is_image_file(fname) and fname in fnames_list:

                path_img = os.path.join(root, fname)
                # print('Path image: ', path_img)

                path_h = os.path.join(dir_h, fname)
                path_v = os.path.join(dir_v, fname)

                path_sem = os.path.join(dir_sem, fname.split('.')[0]+'.jpg')

                path_depth = os.path.join(dir_depth, fname.split('.')[0]+'.png')

                path_feats = os.path.join(dir_feats, fname.split('.')[0]+'.pt')

                images.append([path_img, path_h, path_v, path_sem, path_depth, path_feats])
                #print('paths : ', path_sem, path_depth)
                #if len(images) >= 300:
                #    break

    # print('images: ', len(images))
    return images[:min(max_dataset_size, len(images))]

def default_loader(path):
    return Image.open(path).convert('RGB')

class ImageFolder(data.Dataset):

    def __init__(self, root, num_workers=0, transform=None, return_paths=False,
                 loader=default_loader):
        print('!!!!!!!!!!!!!!!!!!!!!!!In image folder hereee!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        imgs = make_dataset(root)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in: " + root + "\n"
                               "Supported image extensions are: " +
                               ",".join(IMG_EXTENSIONS)))

        self.root = root
        self.imgs = imgs
        self.transform = transform
        self.return_paths = return_paths
        self.loader = loader

    def __getitem__(self, index):
        path = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.return_paths:
            return img, path
        else:
            return img

    def __len__(self):
        return len(self.imgs)
