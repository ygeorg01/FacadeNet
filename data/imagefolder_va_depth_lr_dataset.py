import random
from data.base_dataset import BaseDataset, get_transform
# from data.image_va_depth_sem_folder import make_dataset
from data.image_va_depth_lr_folder import make_dataset
from PIL import Image
import os
import torch
import torchvision.transforms as T
import numpy as np
from torchvision.utils import draw_segmentation_masks

class ImageFolderVaDepthLrDataset(BaseDataset):
    def __init__(self, opt):
        BaseDataset.__init__(self, opt)
        print('Opt in dataset: ', opt)
        self.dir_A = opt.dataroot
        self.dir_h = opt.dataroot_h
        self.dir_v = opt.dataroot_v
        self.dir_sem = opt.dataroot_sem
        self.dir_depth = opt.dataroot_depth
        self.dir_feats = opt.dataroot_feats
        self.load_size = opt.load_size
        self.checkpoint_dir = opt.checkpoints_dir
        self.exp_name = opt.name
        # self.phase = opt.phase

        self.train_files = []
        with open(opt.train_split_dir, 'r') as f:
            lines = f.readlines()
            for line in lines:
                self.train_files.append(line.strip())

        self.eval_files = []
        with open(opt.eval_split_dir, 'r') as f:
            lines = f.readlines()
            for line in lines:
                self.eval_files.append(line.strip())

        if opt.phase == 'train':
            self.A_paths = make_dataset(self.dir_A, self.dir_h, self.dir_v, self.dir_sem, self.dir_depth, self.dir_feats, self.train_files)

        elif opt.phase == 'test':
            self.A_paths = make_dataset(self.dir_A, self.dir_h, self.dir_v, self.dir_sem, self.dir_depth, self.dir_feats, self.eval_files)

        self.A_size = len(self.A_paths)
        self.transform_A = get_transform(self.opt, params={'flip':False}, make_power=True, grayscale=False)

        resize = T.Resize((self.opt.load_size, self.opt.load_size))

        self.imgs = []
        for counter, (path_img, path_h, path_v, path_sem, path_depth, path_feats) in enumerate(self.A_paths):
            self.imgs.append(self.getitem_by_path(path_img, path_h, path_v, path_sem, path_depth, path_feats, self.opt.load_size))

    def get_params(self, opt, size):
        w, h = size
        new_h = h
        new_w = w
        if opt.preprocess == 'resize_and_crop':
            new_h = new_w = opt.load_size
        elif opt.preprocess == 'scale_width_and_crop':
            new_w = opt.load_size
            new_h = opt.load_size * h // w
        x = random.randint(0, np.maximum(0, new_w - opt.crop_size))
        y = random.randint(0, np.maximum(0, new_h - opt.crop_size))

        flip = random.random() > 2

        return {'crop_pos': (x, y), 'flip': flip, 'grayscale':False}

    def crop_tensor(self, img, pos, size):
        ow, oh = img.shape[1:]
        x1, y1 = pos
        tw = th = size
        if (ow > tw or oh > th):
            return img[:, y1:y1 + th, x1:x1 + tw]
        return img
    
    def __getitem__(self, index):

        # Apply flip
        imgs_dict = self.imgs[index % self.A_size]

        imgs = imgs_dict['real_A']
        h = imgs_dict['h']
        v = imgs_dict['v']
        sem = imgs_dict['sem']
        depth = imgs_dict['depth']
        feats = imgs_dict['feats']

        tranform_params = self.get_params(self.opt, imgs.size)
        # tranform_params_va = tranform_params.copy()
        A_transform = get_transform(self.opt, params=tranform_params, convert=True)
        tranform_params['grayscale'] = True
        B_transform = get_transform(self.opt, params=tranform_params, grayscale=True, convert=True)

        imgs = A_transform(imgs)

        sem = B_transform(sem)

        sem = abs((((sem + 1)/2) - 1))

        h = B_transform(h)
        v = B_transform(v)
        depth = B_transform(depth)

        feats = torch.from_numpy(feats).permute(2,0,1).unsqueeze(0)
        # feats = feats.permute(2,0,1).unsqueeze(0)
        min_f = torch.min(torch.flatten(feats, start_dim=2, end_dim=3), dim=-1)[0].unsqueeze(-1).unsqueeze(-1)
        max_f = torch.max(torch.flatten(feats, start_dim=2, end_dim=3), dim=-1)[0].unsqueeze(-1).unsqueeze(-1)

        feats = (feats - min_f) / (max_f - min_f) 
        feats = 1 - feats 

        # from torchvision.utils import save_image
        # for ch in range(4):
        #     save_image(feats[0,ch], 'feats_'+str(ch)+'_.png')

        feats = torch.nn.functional.interpolate(feats, size=(256,256), mode='bilinear').squeeze()
        feats = self.crop_tensor(feats, tranform_params['crop_pos'], 256)

        imgs_dict['real_A'] = imgs

        imgs_dict['v'] = v
        imgs_dict['h'] = h

        kernel = torch.ones((1,1,5,5))

        imgs_dict['sem'] = sem
        imgs_dict['depth'] = depth
        imgs_dict['feats'] = feats

        return imgs_dict


    def getitem_by_path(self, A_path, path_h, path_v, path_sem, path_depth, path_feats, load_size):

        # print(A_path, path_feats)
        A_img = Image.open(A_path).convert('RGB')
        A_img = A_img.resize((load_size, load_size), resample=Image.BICUBIC)

        horizontal_img = Image.open(path_h)
        horizontal_img = horizontal_img.resize((load_size, load_size), resample=Image.BICUBIC)

        vertical_img = Image.open(path_v)
        vertical_img = vertical_img.resize((load_size, load_size), resample=Image.BICUBIC)

        semantic_img = Image.open(path_sem)
        semantic_img = semantic_img.resize((load_size, load_size), resample=Image.BICUBIC)

        depth_img = Image.open(path_depth)
        depth_img = depth_img.resize((load_size, load_size), resample=Image.BICUBIC)

        feats = torch.load(path_feats)

        return {'real_A':A_img, 'path_A':A_path, 'h':horizontal_img, 'h_path':path_h, 'v':vertical_img, 'v_path':path_v
                ,'sem':semantic_img, 'sem_path':path_sem, 'depth':depth_img, 'path_depth':path_depth, 'feats':feats, 'feats_path':path_feats}

    def __len__(self):
        return self.A_size
