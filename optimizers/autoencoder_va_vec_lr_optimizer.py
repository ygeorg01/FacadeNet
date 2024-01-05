import torch
import util
from models import MultiGPUModelWrapper
from optimizers.base_optimizer import BaseOptimizer
import torch.nn.functional as F


class AutoencoderVaVecLrOptimizer(BaseOptimizer):
    """ Class for running the optimization of the model parameters.
    Implements Generator / Discriminator training, R1 gradient penalty,
    decaying learning rates, and reporting training progress.
    """
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument("--lr", default=0.002, type=float)
        parser.add_argument("--beta1", default=0.0, type=float)
        parser.add_argument("--beta2", default=0.99, type=float)
        parser.add_argument(
            "--R1_once_every", default=16, type=int,
            help="lazy R1 regularization. R1 loss is computed "
                 "once in 1/R1_freq times",
        )
        return parser

    def __init__(self, model: MultiGPUModelWrapper):
        self.opt = model.opt
        opt = self.opt

        self.model = model
        self.train_mode_counter = 0
        self.discriminator_iter_counter = 0

        # Gparams have both encoder and decoder parameters
        self.Gparams = self.model.get_parameters_for_mode("generator")
        # Dparam have both D and Dpatch parameters
        self.Dparams = self.model.get_parameters_for_mode("discriminator")

        self.optimizer_G = torch.optim.Adam(
            self.Gparams, lr=opt.lr, betas=(opt.beta1, opt.beta2)
        )

        # c.f. StyleGAN2 (https://arxiv.org/abs/1912.04958) Appendix B
        c = opt.R1_once_every / (1 + opt.R1_once_every)
        self.optimizer_D = torch.optim.Adam(
            self.Dparams, lr=opt.lr * c, betas=(opt.beta1 ** c, opt.beta2 ** c)
        )

        self.blend_weights = torch.full((1, 4, 1, 1), 0.25, device='cuda', requires_grad=True)
        self.sigmoid = torch.nn.Sigmoid()
        self.optimizer_mask = torch.optim.Adam(
            [self.blend_weights], lr=0.01
        )
        self.mse_loss = torch.nn.MSELoss(reduction='mean')
    def set_requires_grad(self, params, requires_grad):
        """ For more efficient optimization, turn on and off
            recording of gradients for |params|.
        """
        for p in params:
            p.requires_grad_(requires_grad)

    def prepare_images(self, data_i):

        return data_i["real_A"], data_i['h'], data_i['v'], data_i["depth"], data_i["feats"]

    def toggle_training_mode(self):
        modes = ["discriminator", "generator"]
        self.train_mode_counter = (self.train_mode_counter + 1) % len(modes)
        return modes[self.train_mode_counter]

    def train_one_step(self, data_i, total_steps_so_far):
        # Load Batch
        images_minibatch, h_minibatch, v_minibatch, depth_minibatch, feats_minibatch = self.prepare_images(data_i)

        real_pred = None
        rec_pred = None

        # Compute Discriminator Loss
        if self.toggle_training_mode() == "generator":
            losses, real_pred, rec_pred  = self.train_discriminator_one_step(images_minibatch, h_minibatch, v_minibatch, depth_minibatch)
        else:
        # Compute Generator Loss
            losses = self.train_generator_one_step(images_minibatch, h_minibatch, v_minibatch, depth_minibatch, feats_minibatch)
        return util.to_numpy(losses), real_pred, rec_pred

    def train_generator_one_step(self, images, h, v, depth, feats):
        self.set_requires_grad(self.Dparams, False)
        self.set_requires_grad(self.Gparams, True)
        sp_ma, gl_ma = None, None

        self.optimizer_G.zero_grad()
        self.optimizer_mask.zero_grad()

        # print('Compute editing mask: ', self.blend_weights, feats.shape)
        mask = torch.sum(self.sigmoid(self.blend_weights) * feats.to('cuda'), dim=1, keepdim=True)
        #mask = F.normalize(mask)
        self.model.blend_weights = self.blend_weights

        g_losses, g_metrics, m_losses, m_metrics = self.model(
            images, h, v, sp_ma, gl_ma, mask, depth, command="compute_generator_losses"
        )

        g_loss = sum([v.mean() for v in g_losses.values()])
        g_loss.backward()
        self.optimizer_G.step()

        # Optimize signal for blending layer
        mask_loss = m_losses['G_sel_L1'].mean() + 10 * (torch.abs(torch.sum(F.relu(self.blend_weights.squeeze()), dim=0)
                                                                  - 1))

        mask_loss.backward()
        self.optimizer_mask.step()

        g_losses.update(g_metrics)
        g_losses.update(m_metrics)


        return g_losses

    def train_discriminator_one_step(self, images, h, v, depth):

        if self.opt.lambda_GAN == 0.0 and self.opt.lambda_PatchGAN == 0.0:
            return {}
        self.set_requires_grad(self.Dparams, True)
        self.set_requires_grad(self.Gparams, False)
        self.discriminator_iter_counter += 1
        self.optimizer_D.zero_grad()
        # d_losses, d_metrics, sp, gl = self.model(
        #     images, command="compute_discriminator_losses"
        # )
        d_losses, d_metrics, sp, real_pred, rec_pred, mix_pred = self.model(
            images, h, v, depth, command="compute_discriminator_losses"
        )
        # if sp != None:
        #     self.previous_sp = sp.detach()
        # self.previous_gl = gl.detach()
        d_loss = sum([v.mean() for v in d_losses.values()])
        d_loss.backward()
        self.optimizer_D.step()

        needs_R1 = self.opt.lambda_R1 > 0.0 or self.opt.lambda_patch_R1 > 0.0
        needs_R1_at_current_iter = needs_R1 and \
            self.discriminator_iter_counter % self.opt.R1_once_every == 0
        if needs_R1_at_current_iter:
            self.optimizer_D.zero_grad()
            self.optimizer_D.zero_grad()
            r1_losses = self.model(images, h, v, command="compute_R1_loss")
            d_losses.update(r1_losses)
            r1_loss = sum([v.mean() for v in r1_losses.values()])
            r1_loss = r1_loss * self.opt.R1_once_every
            r1_loss.backward()
            self.optimizer_D.step()

        d_losses["D_total"] = sum([v.mean() for v in d_losses.values()])
        d_losses.update(d_metrics)
        return d_losses, real_pred, rec_pred

    def get_visuals_for_snapshot(self, data_i):
        images, h, v, depth, feats = self.prepare_images(data_i)

        with torch.no_grad():
            mask = torch.sum(self.sigmoid(self.blend_weights) * feats.to('cuda'), dim=1, keepdim=True)
            #mask = F.normalize(mask)
            return self.model(images, h[:,:,0,:].squeeze(), v[:,:,:,0].squeeze(), depth, mask, command="get_visuals_for_snapshot")

    def save(self, total_steps_so_far):
        self.model.save(total_steps_so_far)
