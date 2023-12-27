import torch
import util
from models import BaseModel
import models.networks as networks
import models.networks.loss as loss
from torchvision.utils import save_image
import random

class AutoencoderVaVecLrModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        BaseModel.modify_commandline_options(parser, is_train)
        parser.add_argument("--spatial_code_ch", default=16, type=int)
        parser.add_argument("--global_code_ch", default=2048, type=int)
        parser.add_argument("--lambda_R1", default=10.0, type=float)
        parser.add_argument("--lambda_patch_R1", default=1.0, type=float)
        parser.add_argument("--lambda_L1", default=1.0, type=float)
        parser.add_argument("--lambda_GAN", default=1.0, type=float)
        parser.add_argument("--lambda_PatchGAN", default=1.0, type=float)
        parser.add_argument("--patch_min_scale", default=1 / 8, type=float)
        parser.add_argument("--patch_max_scale", default=1 / 4, type=float)
        parser.add_argument("--patch_num_crops", default=8, type=int)
        parser.add_argument("--patch_use_aggregation",type=util.str2bool, default=True)
        return parser

    def initialize(self):
        self.E = networks.create_network(self.opt, self.opt.netE, "encoder_nostyle")
        self.G = networks.create_network(self.opt, self.opt.netG, "generator_va_vector")
        if self.opt.lambda_GAN > 0.0:
            self.D = networks.create_network(
                self.opt, self.opt.netD, "discriminator_va")

        # Count the iteration count of the discriminator
        # Used for lazy R1 regularization (c.f. Appendix B of StyleGAN2)
        self.register_buffer(
            "num_discriminator_iters", torch.zeros(1, dtype=torch.long)
        )
        self.l1_loss = torch.nn.L1Loss()

        if (not self.opt.isTrain) or self.opt.continue_train:
            self.load()

        if self.opt.num_gpus > 0:
            self.to("cuda:0")

    def per_gpu_initialize(self):
        pass

    def swap(self, x):
        """ Swaps (or mixes) the ordering of the minibatch """
        shape = x.shape
        assert shape[0] % 2 == 0, "Minibatch size must be a multiple of 2"
        new_shape = [shape[0] // 2, 2] + list(shape[1:])
        x = x.view(*new_shape)
        x = torch.flip(x, [1])
        return x.view(*shape)

    def random_step(self, x, range=0.6, min=-1, max=1, dim_=0):

        interpolation_range_h = range

        dif_min = torch.min(x, dim=dim_)[0][:, 0, 0] - (-interpolation_range_h)

        x = x - dif_min.unsqueeze(1).unsqueeze(1).unsqueeze(1)
        add_margin = abs((torch.max(x, dim=dim_)[0][:, 0, 0] - interpolation_range_h))

        inter = torch.rand(x.shape[0], 1, 1, 1).to(x.device)

        x = x + (inter * add_margin.unsqueeze(1).unsqueeze(1).unsqueeze(1))

        return x

    def interpolated_h_maps(self, h):
        # Horizontal Interpolation
        interpolation_range = 0.95
        h_steps = 200

        h = ((h - torch.amin(h, dim=(2, 3)).unsqueeze(1).unsqueeze(1)) / (
                    torch.amax(h, dim=(2, 3)) - torch.amin(h, dim=(2, 3))).unsqueeze(1).unsqueeze(1)) * 0.95

        diff = torch.amin(h, dim=(2, 3)) + interpolation_range

        h = h - diff.unsqueeze(1).unsqueeze(1)

        add_value_h = abs((torch.amax(h, dim=(2, 3)) - interpolation_range) / h_steps)

        steps = torch.randint(h_steps, (h.shape[0],)).to('cuda')

        # addition = torch.matmul(steps, add_value_h)
        addition = steps.unsqueeze(1) * add_value_h

        return h + addition.unsqueeze(1).unsqueeze(1)

    def compute_image_discriminator_losses(self, real, rec, mix, h, h_swapped, v, v_swapped):

        if self.opt.lambda_GAN == 0.0:
            return {}

        real_input = torch.cat((real, h, v), dim=1)
        pred_real, features_real = self.D(real_input)

        rec_input = torch.cat((rec, h, v), dim=1)
        pred_rec, features_rec = self.D(rec_input)

        mix_input = torch.cat((mix, h_swapped, v_swapped), dim=1)
        pred_mix, features_mix = self.D(mix_input)



        losses = {}
        losses["D_real"] = loss.gan_loss(
            pred_real, should_be_classified_as_real=True
        ) * self.opt.lambda_GAN

        losses["D_rec"] = loss.gan_loss(
            pred_rec, should_be_classified_as_real=False
        ) * (0.5 * self.opt.lambda_GAN)

        losses["D_mix"] = loss.gan_loss(
            pred_mix, should_be_classified_as_real=False
        ) * (0.5 * self.opt.lambda_GAN)

        return losses, pred_real, pred_rec, pred_mix

    def get_random_crops(self, x, crop_window=None):
        """ Make random crops.
            Corresponds to the yellow and blue random crops of Figure 2.
        """
        crops = util.apply_random_crop(
            x, self.opt.patch_size,
            (self.opt.patch_min_scale, self.opt.patch_max_scale),
            num_crops=self.opt.patch_num_crops
        )
        return crops

    def compute_patch_discriminator_losses(self, real, mix):
        losses = {}
        real_feat = self.Dpatch.extract_features(
            self.get_random_crops(real),
            aggregate=self.opt.patch_use_aggregation
        )
        target_feat = self.Dpatch.extract_features(self.get_random_crops(real))
        mix_feat = self.Dpatch.extract_features(self.get_random_crops(mix))

        losses["PatchD_real"] = loss.gan_loss(
            self.Dpatch.discriminate_features(real_feat, target_feat),
            should_be_classified_as_real=True,
        ) * self.opt.lambda_PatchGAN

        losses["PatchD_mix"] = loss.gan_loss(
            self.Dpatch.discriminate_features(real_feat, mix_feat),
            should_be_classified_as_real=False,
        ) * self.opt.lambda_PatchGAN

        return losses

    def compute_discriminator_losses(self, real, h, v, depth):

        self.num_discriminator_iters.add_(1)

        # real_e_input = torch.cat((real, depth), axis=1)
        sp = self.E(real)
        B = real.size(0)
        assert B % 2 == 0, "Batch size must be even on each GPU."

        # To save memory, compute the GAN loss on only
        # half of the reconstructed images

        h_vec = h[:,:,0,:].squeeze()
        v_vec = v[:,:,:,0].squeeze()

        rec = self.G(sp, h_vec, v_vec)

        # h_swapped = self.interpolated_h_maps(h.clone())
        # h_swapped = self.swap(h.clone())
        swap_v = False
        if torch.rand(1) > 0.5:
            v_swapped = self.swap(v.clone())
            swap_v = True
        else:
            v_swapped = v.clone()

        if torch.rand(1) > 0.5:
            h_swapped = self.random_step(h.clone(), min=torch.min(h, dim=3)[0][:,0,0], max=torch.max(h, dim=3)[0][:,0,0], dim_=3)
        else:
            h_swapped = self.swap(h.clone())

        h_swapped_vec = h_swapped[:, :, 0, :].squeeze()
        v_swapped_vec = v_swapped[:, :, :, 0].squeeze()

        mix = self.G(sp, h_swapped_vec, v_swapped_vec)

        losses, pred_real, pred_rec, pred_mix = self.compute_image_discriminator_losses(real, rec, mix, h, h_swapped, v, v_swapped)

        metrics = {}  # no metrics to report for the Discriminator iteration

        return losses, metrics, sp.detach(), pred_real, pred_rec, pred_mix#, gl.detach()

    def compute_R1_loss(self, real, h, v):
        losses = {}
        if self.opt.lambda_R1 > 0.0:
            real.requires_grad_()
            input_ = torch.cat((real, h, v), dim=1)
            pred_real = self.D(input_)[0].sum()
            grad_real, = torch.autograd.grad(
                outputs=pred_real,
                inputs=[input_],
                create_graph=True,
                retain_graph=True,
            )
            grad_real2 = grad_real.pow(2)
            dims = list(range(1, grad_real2.ndim))
            grad_penalty = grad_real2.sum(dims) * (self.opt.lambda_R1 * 0.5)
        else:
            grad_penalty = 0.0

        losses["D_R1"] = grad_penalty

        return losses

    def compute_generator_losses(self, real, h, v, sp_ma, gl_ma, sem, depth):

        losses, metrics = {}, {}

        losses_mask, metrics_mask = {}, {}

        B = real.size(0)

        # real_e_input = torch.cat((real, depth), dim=1)
        sp = self.E(real)
        h_vec = h[:,:,0,:].squeeze()
        v_vec = v[:,:,:,0].squeeze()

        rec = self.G(sp, h_vec, v_vec)  # only on B//2 to save memory

        metrics["L1_dist_novel"] = 0
        metrics_mask["L1_dist_sel"] = 0

        nvs_counter = 1
        for nvs in range(nvs_counter):
            swap_v = False
            if torch.rand(1) > 0.5:
                v_swapped = self.swap(v.clone())
                swap_v = True
            else:
                v_swapped = v.clone()

            if torch.rand(1) > 0.5 or not swap_v:
                h_swapped = self.random_step(h.clone(), min=torch.min(h, dim=3)[0][:,0,0], max=torch.max(h, dim=3)[0][:,0,0], dim_=3)
            else:
                h_swapped = h.clone()

            h_swapped_vec = h_swapped[:, :, 0, :].squeeze()
            v_swapped_vec = v_swapped[:, :, :, 0].squeeze()
            mix = self.G(sp, h_swapped_vec, v_swapped_vec)

            # print('Semantics shape: ', sem.shape)
            metrics["L1_dist_novel"] += self.l1_loss(mix * sem.detach(), real * sem.detach())
            metrics_mask["L1_dist_sel"] += self.l1_loss(mix.detach() * sem, real.detach() * sem)

        # record the error of the reconstructed images for monitoring purposes
        metrics["L1_dist"] = self.l1_loss(rec, real)

        if self.opt.lambda_L1 > 0.0:
            losses["G_L1"] = metrics["L1_dist"] * self.opt.lambda_L1

        if self.opt.lambda_sel_L1 > 0.0:
            losses["G_sel_L1"] = (metrics["L1_dist_novel"] * self.opt.lambda_sel_L1) / nvs_counter
            losses_mask["G_sel_L1"] = metrics_mask["L1_dist_sel"] / nvs_counter

        rec_input = torch.cat((rec, h, v), dim=1)
        pred_rec, features_rec = self.D(rec_input)

        mix_input = torch.cat((mix, h_swapped, v_swapped), dim=1)
        pred_mix, features_mix = self.D(mix_input)
        if self.opt.lambda_GAN > 0.0:
            losses["G_GAN_rec"] = loss.gan_loss(
                pred_rec,
                should_be_classified_as_real=True
            ) * (self.opt.lambda_GAN * 0.5)

            losses["G_GAN_mix"] = loss.gan_loss(
                pred_mix,
                should_be_classified_as_real=True
            ) * (self.opt.lambda_GAN * 0.5)

        return losses, metrics, losses_mask, metrics_mask

    def get_visuals_for_snapshot(self, real, h, v, depth, mask):

        # save_image(mask, 'mask.png')

        if self.opt.isTrain:
            # avoid the overhead of generating too many visuals during training
            real = real[:2] if self.opt.num_gpus > 1 else real[:4]
            h = h[:2] if self.opt.num_gpus > 1 else h[:4]
            v = v[:2] if self.opt.num_gpus > 1 else v[:4]
            depth = depth[:2] if self.opt.num_gpus > 1 else depth[:4]
            mask = mask[:2] if self.opt.num_gpus > 1 else mask[:4]

        sp = self.E(real)

        # Create an empty  spatial code
        layout = util.resize2d_tensor(util.visualize_spatial_code(sp), real)
        rec = self.G(sp, h, v)

        visuals = {"real": real, "layout": layout, "rec": rec, "mask": mask, "mix": None}

        return visuals

    def fix_noise(self, sample_image=None):
        """ The generator architecture is stochastic because of the noise
        input at each layer (StyleGAN2 architecture). It could lead to
        flickering of the outputs even when identical inputs are given.
        Prevent flickering by fixing the noise injection of the generator.
        """
        if sample_image is not None:
            # The generator should be run at least once,
            # so that the noise dimensions could be computed
            sp, gl = self.E(sample_image)
            self.G(sp, gl)
        noise_var = self.G.fix_and_gather_noise_parameters()
        return noise_var

    def encode(self, image, extract_features=False):
        return self.E(image, extract_features=extract_features)

    def decode(self, spatial_code, h, v):

        return self.G(spatial_code, h, v)

    def get_parameters_for_mode(self, mode):
        if mode == "generator":
            return list(self.G.parameters()) + list(self.E.parameters())
        elif mode == "discriminator":
            Dparams = []
            if self.opt.lambda_GAN > 0.0:
                Dparams += list(self.D.parameters())
                # Dparams += list(self.D_va.parameters())
            # if self.opt.lambda_PatchGAN > 0.0:
            #     Dparams += list(self.Dpatch.parameters())
            return Dparams
