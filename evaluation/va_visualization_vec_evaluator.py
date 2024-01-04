import os
from PIL import Image
import numpy as np
import torch
from evaluation import BaseEvaluator
import util
from util.evaluator import PSNR, SSIM
from util.fid_score import calculate_fid_given_images
from torchvision.utils import save_image
import imageio
import ignite.metrics as metrics
from skimage import img_as_ubyte
import lpips
from cleanfid import fid

class VaVisualizationVecEvaluator(BaseEvaluator):

    def __init__(self, opt, target_phase):
        BaseEvaluator.__init__(self, opt, target_phase)

        self.psnr = metrics.PSNR(data_range=1.0, device='cuda')
        self.ssim = metrics.SSIM(data_range=1.0, device='cuda')
        # self.fid_rec = metrics.FID(device='cuda')
        # self.fid_random = metrics.FID(device='cuda')
        self.batch_size = opt.batch_size
        self.loss_fn_vgg = lpips.LPIPS(net='vgg').to('cuda')  # closer to "traditional" perceptual loss, when used for optimization
        self.loss_fn_alex = lpips.LPIPS(net='alex').to('cuda')  # closer to "traditional" perceptual loss, when used for optimization
        self.sigmoid = torch.nn.Sigmoid()

    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument("--swap_num_columns", type=int, default=4,
                            help="number of images to be shown in the swap visualization grid. Setting this value will result in 4x4 swapping grid, with additional row and col for showing original images.")
        parser.add_argument("--swap_num_images", type=int, default=16,
                            help="total number of images to perform swapping. In the end, (swap_num_images / swap_num_columns) grid will be saved to disk")
        return parser

    def gather_images(self, dataset):
        all_images = []
        all_h = []
        all_v = []

        num_images_to_gather = max(self.opt.swap_num_columns, self.opt.num_gpus)
        exhausted = False

        while len(all_images) < num_images_to_gather:
            try:
                data = next(dataset)
            except StopIteration:
                print("Exhausted the dataset at %s" % (self.opt.dataroot))
                exhausted = True
                break
            for i in range(data["real_A"].size(0)):
                all_images.append(data["real_A"][i:i+1])
                all_h.append(data["h"][i:i+1])
                all_v.append(data["v"][i:i+1])

                # all_vas_center.append(data["va_center"][i:i+1])
                if "real_B" in data:
                    all_images.append(data["real_B"][i:i+1])
                if len(all_images) >= num_images_to_gather:
                    break
        if len(all_images) == 0:
            return None, None, True, False

        return all_images, all_h, all_v, exhausted

    def generate_samples(self, model, images, hs, vs):
        sps, gls = [], []
        for image in images:
            assert image.size(0) == 1
            sp = model(image.expand(self.opt.num_gpus, -1, -1, -1), command="encode")
            sp = sp[:1]

            sps.append(sp)
        
        def put_img(img, canvas, row, col):
            h, w = img.shape[0], img.shape[1]
            start_x = int(self.opt.crop_size * col + (self.opt.crop_size - w) * 0.5)
            start_y = int(self.opt.crop_size * row + (self.opt.crop_size - h) * 0.5)
            canvas[start_y:start_y + h, start_x: start_x + w, :] = img

        # Create viewing angle samples
        grid_w = self.opt.crop_size * (len(images)+1)  # 2 Columns
        grid_h = self.opt.crop_size * (len(images)+1)

        grid_va_v = np.ones((grid_h, grid_w, 3), dtype=np.uint8)
        grid_va_h = np.ones((grid_h, grid_w, 3), dtype=np.uint8)

        # grid_va_h = np.ones((grid_h, grid_w, 3), dtype=np.uint8)

        for i, (img, h, v) in enumerate(zip(images, hs, vs)):

            image_np = util.tensor2im(img, tile=False)[0]
            put_img(image_np, grid_va_v, i+1, 0)
            put_img(image_np, grid_va_h, i + 1, 0)

            va_v = v.repeat(1, 3, 1, 1)
            va_v = util.tensor2im(va_v.squeeze(), tile=False)
            put_img(va_v, grid_va_v, 0, i + 1)

            va_h = h.repeat(1, 3, 1, 1)
            va_h = util.tensor2im(va_h.squeeze(), tile=False)
            put_img(va_h, grid_va_h, 0, i + 1)

        for i, sp in enumerate(sps):
            # sp = sp.repeat(2, 1, 1, 1)
            for j, (h, v) in enumerate(zip(hs, vs)):

                # Change vertical va map
                # va_0 = va[:, 0, :, :].unsqueeze(1)

                # va_1 = va_maps[i][:, 1, :, :].unsqueeze(1)
                # va = torch.cat((va_0, va_1), dim=1)

                fake_img = model(sp, h, v, command="decode")

                fake_img = util.tensor2im(fake_img, tile=False)

                put_img(fake_img[0].squeeze(), grid_va_v, i+1, j+1)

        grid_v = Image.fromarray(grid_va_v)

        for i, sp in enumerate(sps):
            # sp = sp.repeat(2, 1, 1, 1)
            for j, (h, v) in enumerate(zip(hs, vs)):

                fake_img = model(sp, h, v, command="decode")

                fake_img = util.tensor2im(fake_img, tile=False)

                put_img(fake_img[0].squeeze(), grid_va_h, i+1, j+1)

        grid_h = Image.fromarray(grid_va_h)

        return None, grid_v, grid_h

    def new_mask(self, x, mean=0):

        input_mean = torch.mean(x)
        diff = mean - input_mean

        return x + diff

    def evaluate_metrics(self, model, dataset, nsteps):

        savedir = os.path.join(self.output_dir(), "%s_%s" % (self.target_phase, nsteps))
        os.makedirs(savedir, exist_ok=True)

        fid_src_dir = os.path.join(savedir,'fid_src')
        os.makedirs(fid_src_dir, exist_ok=True)

        fid_rec_dir = os.path.join(savedir,'fid_rec')
        os.makedirs(os.path.join(savedir,'fid_rec'), exist_ok=True)

        mask_rec_dir = os.path.join(savedir,'mask_src')
        os.makedirs(os.path.join(savedir,'mask_src'), exist_ok=True)

        fid_rand_dir = os.path.join(savedir,'fid_rand')
        os.makedirs(os.path.join(savedir,'fid_rand'), exist_ok=True)

        self.psnr.reset()
        self.ssim.reset()

        count_saved_images = 0
        count_rand_images = 0
        LPIPS_vgg = 0.0
        LPIPS_alex = 0.0
        print('Eval dataset length: ', len(dataset))
        with torch.no_grad():
            img_count = 0
            batch_count = 0
            for idx, data in enumerate(dataset):
                if data['real_A'].shape[0]!=self.batch_size:
                    continue

                imgs = data['real_A']
                feats = data['feats']
                img_count += imgs.shape[0]
                batch_count += 1

                mask = self.sigmoid(torch.sum(model.blend_weights * feats.to('cuda'), dim=1, keepdim=True))
                # mask = F.normalize(mask)

                sp = model(imgs, command="encode")

                # Prepare v & h vectors
                v = data['v']
                v = v[:, :, :, 0].squeeze()
                v_vec = v.clone()
                
                h = data['h']
                h = h[:, :, 0, :].squeeze()
                h_vec = h.clone()

                gen_imgs = model(sp, h_vec, v_vec, command="decode")

                gen_imgs = torch.clip(gen_imgs, min=-1, max=1)
                gen_imgs_norm = (gen_imgs + 1)/2
                imgs_norm = (imgs + 1)/2

                # Random mask
                # [-40, 0, 40]
                for ii in range(imgs.shape[0]):

                    img_new = imgs[ii].unsqueeze(0)
                    # Horizontal interpolation for evaluation
                    for mean in [-0.4, 0., 0.4]:
                    # for mean in [-0.3,  -0.15, 0., 0.15, 0.3]:
                    # for mean in [-0.3, -0.275, -0.25, -0.225, -0.2, -0.175, -0.15, -0.125, -0.1, -0.075, -0.05,

                        h_new = self.new_mask(h[ii].unsqueeze(0), mean=mean)
                        h_new = h_new.repeat(self.opt.num_gpus, 1)
                        v_new = v_vec[ii].unsqueeze(0)
                        v_new = v_new.repeat(self.opt.num_gpus, 1)

                        sp_new = sp[ii].unsqueeze(0)
                        sp_new = sp_new.repeat(self.opt.num_gpus, 1, 1, 1)


                        new_imgs = model(sp_new, h_new, v_new, command="decode")

                        new_imgs = torch.clip(new_imgs, min=-1, max=1)
                        new_imgs_norm = (new_imgs + 1) / 2
                        save_image(new_imgs_norm[0],
                                   os.path.join(fid_rand_dir, str(count_rand_images).zfill(5) + '.png'))
                        count_rand_images+=1

                        LPIPS_vgg += self.loss_fn_vgg(img_new[0].to(gen_imgs.device).unsqueeze(0), new_imgs[0].unsqueeze(0)).squeeze()
                        LPIPS_alex += self.loss_fn_alex(img_new[0].to(gen_imgs.device).unsqueeze(0), new_imgs[0].unsqueeze(0)).squeeze()

                self.psnr.update([gen_imgs_norm, imgs_norm.to(gen_imgs.device)]) 
                self.ssim.update([gen_imgs_norm, imgs_norm.to(gen_imgs.device)])

                for ii in range(gen_imgs.shape[0]):
                    # Store images

                    save_image(gen_imgs_norm[ii].unsqueeze(0), os.path.join(fid_rec_dir, str(count_saved_images).zfill(5)+'.png'))
                    save_image(imgs_norm[ii].unsqueeze(0), os.path.join(fid_src_dir, str(count_saved_images).zfill(5)+'.png'))
                    save_image(mask[ii].unsqueeze(0), os.path.join(mask_rec_dir, str(count_saved_images).zfill(5)+'.png'))
                    count_saved_images+=1


                # im = ((np.clip(np.transpose(fake_img.cpu().detach().numpy().squeeze(), (1, 2, 0)), -1,
                #                1) + 1) * 127.5).astype(np.uint8)

                if img_count >= len(dataset):
                    break

        score_rand = fid.compute_fid(fid_src_dir, fid_rand_dir)
        score_rec = fid.compute_fid(fid_src_dir, fid_rec_dir)

        psnr = self.psnr.compute()
        ssim = self.ssim.compute()

        LPIPS_vgg = (LPIPS_vgg/count_rand_images).item()
        LPIPS_alex = (LPIPS_alex/count_rand_images).item()

        return psnr, ssim, LPIPS_vgg, LPIPS_alex, score_rand, score_rec

    def evaluate(self, model, dataset, nsteps):

        if nsteps != 'latest':
            # if nsteps != '1000k':
            nsteps = self.opt.resume_iter if nsteps is None else str(round(nsteps / 1000)) + "k"
            # nsteps = self.opt.resume_iter if nsteps

        savedir = os.path.join(self.output_dir(), "%s_%s" % (self.target_phase, nsteps))
        os.makedirs(savedir, exist_ok=True)
        webpage_title = "Swap Visualization of %s. iter=%s. phase=%s" % \
                            (self.opt.name, str(nsteps), self.target_phase)
        webpage = util.HTML(savedir, webpage_title)
        num_repeats = int(np.ceil(self.opt.swap_num_images / max(self.opt.swap_num_columns, self.opt.num_gpus)))

        try:
            os.makedirs(os.path.join(savedir, 'source'))
        except:
            print('folder exist')

        try:
            os.makedirs(os.path.join(savedir, 'center'))
        except:
            print('folder exist')

        try:
            os.makedirs(os.path.join(savedir, 'rec'))
        except:
            print('folder exist')

        for i in range(0):

            try:
                # print('Data: ')
                images = next(dataset)
                # print('Data: ', data)
            except StopIteration:
                print("Exhausted the dataset at %s" % (self.opt.dataroot))
                exhausted = True
                break
            # print(images)

            # print('Images in eval: ', images)
            if images is None:
                break
            # mix_grid, grid_v, grid_h = self.generate_samples(model, images, hs, vs)
            # webpage.add_images([mix_grid], ["%04d.png" % i])
            # webpage.add_images([grid_v], ["%04d_vertical.png" % i])
            # webpage.add_images([grid_h], ["%04d_horizontal.png" % i])
            # if should_break:
            #     break

        for iter_ in range(3):
            # print('Iter: ', iter_)
            images, hs, vs, should_break = self.gather_images(dataset)

            try:
                os.makedirs(os.path.join(savedir, str(iter_)))
            except:
                print('folder exist')

            for i, image in enumerate(images):
                # if not os.path.exists(os.path.join(savedir,str(iter_), str(i))):
                save_image((image+1)/2,  os.path.join(savedir, 'source', str(iter_).zfill(5)+".png"))
                break

            for i, (image, h, v) in enumerate(zip(images, hs, vs)):
                # if i>0:
                #     break
                sp = model(image.repeat(self.opt.num_gpus, 1, 1, 1), command="encode")

                # Horizontal Interpolation
                interpolation_range_h = 0.60
                interpolation_range_v = 0.4

                h_steps = 10
                v_steps = 1
                # h = va[:,0,:,:].unsqueeze(1)
                # h = ((h-torch.min(h))/ (torch.max(h)-torch.min(h)) * 2) - 1
                # h * 0.2
                # v_norm = ((v-torch.min(v))/ (torch.max(v)-torch.min(v)) * 2) -1

                #h = ((h - (-1.0)) / (0.73 - (-1.0)) * 2) - 1.27
                #v = ((v - (-1.0)) / (0.1 - (-1.0)) * 2) - 1

                # print('h diff: ', torch.max(h) - torch.min(h))
                # print('v diff: ', torch.max(v) - torch.min(v))

                # print((torch.max(h) - torch.min(h)))
                # h = h_norm * ((torch.max(h) - torch.min(h))/2)
                # v = v_norm * ((torch.max(v) - torch.min(v))/2)
                # print('v norm: ', torch.min(v), torch.max(v))

                # h *= 2

                #print('h min and max before: ', torch.min(h), torch.max(h))

                dif_min = torch.min(h) - (-interpolation_range_h)
                h = h - dif_min

                #print('h min and max: ', torch.min(h), torch.max(h))



                #v *= 0.6
                #dif_min = (torch.min(v) - -interpolation_range_v)
                #v = v - dif_min

                # h = h - 0.1
                # v = v - 0.1

                add_value_h = abs(((torch.max(h) - interpolation_range_h) / h_steps))
                #add_value_v = abs(((torch.max(v) - interpolation_range_v) / v_steps))

                # initial masks
                # h = h - torch.min(h)
                # h = ((h - torch.min(h)) / (torch.max(h) - torch.min(h)) * 2) - 1

                # print('Starting h: ', h)
                # va_1 = va[:,1,:,:].unsqueeze(1)
                # diff = torch.min(h) + interpolation_range
                # diff = h + interpolation_range
                # diff = va_1 + interpolation_range


                # h = h - diff

                h_start = h.clone()
                # print('After subtraction horizontal: ', diff, torch.min(va_1), torch.max(va_1))
                # add_value_h = abs(((torch.max(h) - interpolation_range)/h_steps))
                # add_value_h = abs(((torch.min(h) - interpolation_range)/h_steps))

                # add_value_h = abs(((h - interpolation_range_h)/h_steps))

                # print('Add value')
                inter_images = []
                horizontal_images = []
                vertical_images = []
                interpolation_average = []
                center_image = None

                # h += (add_value_h * 50)

                # h = (h - torch.min(h)) / (torch.max(h) - torch.min(h))



                # fake_img = model(sp, h, v, command="decode")
                #
                # im = ((np.clip(np.transpose(fake_img.cpu().detach().numpy().squeeze(), (1, 2, 0)), -1,
                #                1) + 1) * 127.5).astype(np.uint8)
                #
                # center_image = im
                #


                h_start = h.clone()

                for interpolation_index in range(v_steps+1):
                   h = h_start.clone()
                   # v = torch.clip(v, min=-1, max=0)
                   for interpolation_index in range(h_steps+1):

                       # h = h.repeat(self.opt.num_gpus, 1, 1, 1)
                       # h = torch.clamp(h, min=-1, max=1)
                       # v = v.repeat(self.opt.num_gpus, 1, 1, 1)
                       # v = torch.clamp(v, min=-1, max=1)

                       # print('H shape: ', h)
                       # print('V shape: ', v)
                       # print('sp shape: ', sp)

                       # h = torch.clip(h-0.1, min=-0.55, max=0.55)
                       h_vec = h[:, :, 0, :].squeeze().unsqueeze(0)
                       v_vec = v[:, :, :, 0].squeeze().unsqueeze(0)

                       # v_vec = v_vec.repeat(self.opt.num_gpus, 1)
                       # h_vec = h_vec.repeat(self.opt.num_gpus, 1)
                       # sp = sp.repeat(self.opt.num_gpus, 1, 1, 1)

                       print('h vec: ', torch.min(h_vec), torch.max(h_vec))
                       v_vec = v_vec.repeat(self.opt.num_gpus, 1)
                       h_vec = h_vec.repeat(self.opt.num_gpus, 1)
                       # sp = sp.repeat(self.opt.num_gpus, 1, 1, 1)

                       fake_img = model(sp, h_vec, v_vec, command="decode")

                       im = ((np.clip(np.transpose(fake_img[0].cpu().detach().numpy().squeeze(), (1, 2, 0)), -1, 1)+1)*127.5).astype(np.uint8)
                       h_s = ((np.clip(h[0].cpu().detach().numpy().squeeze(), -1, 1) + 1) * 127.5).astype(np.uint8)
                       v_s = ((np.clip(v[0].cpu().detach().numpy().squeeze(), -1, 1) + 1) * 127.5).astype(np.uint8)

                       inter_images.append(im)
                       horizontal_images.append(h_s)
                       vertical_images.append(v_s)



                       h += add_value_h
                   v += 0.02

                       # if interpolation_index == 30:
                       #     center_image=im



                # print('Average: ', sum(interpolation_average)/len(interpolation_average))
                imageio.mimsave(os.path.join(savedir, str(iter_).zfill(5)+'_h.gif'), inter_images)
                # imageio.imsave(os.path.join(savedir, str(iter_), str(i), 'center.png'), center_image)
                # imageio.imsave(os.path.join(savedir, 'center', str(iter_).zfill(5)+'.png'), center_image)

                for i, (im, h, v) in enumerate(zip(inter_images, horizontal_images, vertical_images)):
                    imageio.imsave(os.path.join(savedir, str(iter_), str(i).zfill(5) + '_img.png'), im)
                    # imageio.imsave(os.path.join(savedir, str(iter_), str(i).zfill(5)+'_h.png'), h)
                    # imageio.imsave(os.path.join(savedir, str(iter_), str(i).zfill(5) + '_v.png'), v)

                break
                # Vertical Interpolation
                # interpolation_range = 1.0
                # v_steps = 40
                # va_0 = va[:,0,:,:].unsqueeze(1)
                # # va_1 = va[:,1,:,:].unsqueeze(1)
                # diff = torch.min(va_0) - (-interpolation_range)
                # va_0 = va_0 - diff
                # va_v_start = va_0.clone()
                # # print('After subtraction vertical: ', diff, torch.min(va_0), torch.max(va_0))
                # add_value_v = ((interpolation_range - va_0)/v_steps)
                #
                # inter_images = []
                # va_1_center = va_1_center.repeat(self.opt.num_gpus, 1, 1, 1)
                #
                # for interpolation_index in range(v_steps):
                #
                #     # va = torch.cat((va_0, va_1), dim=1)
                #     va = torch.cat((va_0, va_1_center), dim=1)
                #
                #     fake_img = model(sp, va, command="decode")
                #     # print(fake_img.cpu().detach().numpy().squeeze()[[1,2,0]].shape)
                #     inter_images.append(np.transpose(fake_img.cpu().detach().numpy().squeeze(), (1,2,0)))
                #     # fake_img = util.tensor2im(fake_img, tile=False)
                #     # save_image((fake_img+1)/2, os.path.join(self.output_dir(), 'inter_v_'+str(interpolation_index)+'.png'))
                #
                #     if interpolation_index == v_steps/2:
                #         va_0_center = va_0.clone()
                #
                #     va_0 += add_value_v
                #
                # imageio.mimsave(os.path.join(savedir, str(i), 'v.gif'), inter_images)

        webpage.save()
        return {}
