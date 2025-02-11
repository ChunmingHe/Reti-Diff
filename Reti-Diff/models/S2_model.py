import numpy as np
import random
import torch
from basicsr.data.degradations import random_add_gaussian_noise_pt, random_add_poisson_noise_pt
from basicsr.data.transforms import paired_random_crop
from basicsr.models.sr_model import SRModel
from basicsr.utils import DiffJPEG, USMSharp
from basicsr.utils.img_process_util import filter2D
from basicsr.utils.registry import MODEL_REGISTRY
from torch.nn import functional as F
from collections import OrderedDict
from models import lr_scheduler as lr_scheduler
from torch import nn
from basicsr.archs import build_network
from basicsr.utils import get_root_logger
from basicsr.losses import build_loss
import os

class Mixing_Augment:
    def __init__(self, mixup_beta, use_identity, device):
        self.dist = torch.distributions.beta.Beta(torch.tensor([mixup_beta]), torch.tensor([mixup_beta]))
        self.device = device

        self.use_identity = use_identity

        self.augments = [self.mixup]

    def mixup(self, target, input_):
        lam = self.dist.rsample((1,1)).item()
    
        r_index = torch.randperm(target.size(0)).to(self.device)
    
        target = lam * target + (1-lam) * target[r_index, :]
        input_ = lam * input_ + (1-lam) * input_[r_index, :]
    
        return target, input_

    def __call__(self, target, input_):
        if self.use_identity:
            augment = random.randint(0, len(self.augments))
            if augment < len(self.augments):
                target, input_ = self.augments[augment](target, input_)
        else:
            augment = random.randint(0, len(self.augments)-1)
            target, input_ = self.augments[augment](target, input_)
        return target, input_


def get_batchnorm_layer(opts):
    if opts.norm_layer == "batch":
        norm_layer = nn.BatchNorm2d
    elif opts.layer == "spectral_instance":
        norm_layer = nn.InstanceNorm2d
    else:
        print("not implemented")
        exit()
    return norm_layer

def get_conv2d_layer(in_c, out_c, k, s, p=0, dilation=1, groups=1):
    return nn.Conv2d(in_channels=in_c,
                    out_channels=out_c,
                    kernel_size=k,
                    stride=s,
                    padding=p,dilation=dilation, groups=groups)

def get_deconv2d_layer(in_c, out_c, k=1, s=1, p=1):
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode="bilinear"),
        nn.Conv2d(
            in_channels=in_c,
            out_channels=out_c,
            kernel_size=k,
            stride=s,
            padding=p
        )
    )
class Decom(nn.Module):
    def __init__(self):
        super().__init__()
        self.decom = nn.Sequential(
            get_conv2d_layer(in_c=3, out_c=32, k=3, s=1, p=1),
            nn.LeakyReLU(0.2, inplace=True),
            get_conv2d_layer(in_c=32, out_c=32, k=3, s=1, p=1),
            nn.LeakyReLU(0.2, inplace=True),
            get_conv2d_layer(in_c=32, out_c=32, k=3, s=1, p=1),
            nn.LeakyReLU(0.2, inplace=True),
            get_conv2d_layer(in_c=32, out_c=4, k=3, s=1, p=1),
            nn.ReLU()
        )

    def forward(self, input):
        decom = self.decom(input)
        R = decom[:, 0:3, :, :]
        L = decom[:, 3:4, :, :]
        return R, L

def aux_load_initialize(model, decom_model_path):
    if os.path.exists(decom_model_path):
        checkpoint_Decom_low = torch.load(decom_model_path)
        model.load_state_dict(checkpoint_Decom_low['state_dict']['model_R'])
        # to freeze the params of Decomposition Model
        for param in model.parameters():
            param.requires_grad = False
        return model
    else:
        print("pretrained Initialize Model does not exist, check ---> %s " % decom_model_path)
        exit()

@MODEL_REGISTRY.register()
class RetiDiff_S2Model(SRModel):
    """
    It is trained without GAN losses.
    It mainly performs:
    1. randomly synthesize LQ images in GPU tensors
    2. optimize the networks with GAN training.
    """

    def __init__(self, opt):
        super(RetiDiff_S2Model, self).__init__(opt)
        if self.is_train:
            self.mixing_flag = self.opt['train']['mixing_augs'].get('mixup', False)
            if self.mixing_flag:
                print("-----------------------mixup on-----------------------")
                mixup_beta = self.opt['train']['mixing_augs'].get('mixup_beta', 1.2)
                use_identity = self.opt['train']['mixing_augs'].get('use_identity', False)
                self.mixing_augmentation = Mixing_Augment(mixup_beta, use_identity, self.device)
        self.net_g_S1 = build_network(opt['network_S1'])
        self.net_g_S1 = self.model_to_device(self.net_g_S1)

        # load pretrained models
        load_path = self.opt['path'].get('pretrain_network_S1', None)
        if load_path is not None:
            param_key = self.opt['path'].get('param_key_g', 'params')
            self.load_network(self.net_g_S1, load_path, True, param_key)
        
        self.net_g_S1.eval()
        if self.opt['dist']:
            self.model_Es1_rex = self.net_g_S1.module.E_rex
            self.model_Es1_img = self.net_g_S1.module.E_img
        else:
            self.model_Es1_rex = self.net_g_S1.E_rex
            self.model_Es1_img = self.net_g_S1.E_img
        self.pixel_unshuffle = nn.PixelUnshuffle(4)
        if self.is_train:
            self.encoder_iter = opt["train"]["encoder_iter"]
            self.lr_encoder = opt["train"]["lr_encoder"]
            self.lr_sr = opt["train"]["lr_sr"]
            self.gamma_encoder = opt["train"]["gamma_encoder"]
            self.gamma_sr = opt["train"]["gamma_sr"]
            self.lr_decay_encoder = opt["train"]["lr_decay_encoder"]
            self.lr_decay_sr = opt["train"]["lr_decay_sr"]

        self.Decom_l = Decom().cuda()
        self.Decom_l = aux_load_initialize(self.Decom_l, opt['pretrain_decomnet_low'])
        self.Decom_l.eval()

        self.Decom_h = Decom().cuda()
        self.Decom_h = aux_load_initialize(self.Decom_h, opt['pretrain_decomnet_high'])
        self.Decom_h.eval()

    def setup_optimizers(self):
        train_opt = self.opt['train']
        optim_params = []
        for k, v in self.net_g.named_parameters():
            if v.requires_grad:
                optim_params.append(v)
            else:
                logger = get_root_logger()
                logger.warning(f'Params {k} will not be optimized in the second stage.')

        optim_type = train_opt['optim_g'].pop('type')
        self.optimizer_g = self.get_optimizer(optim_type, optim_params, **train_opt['optim_g'])
        self.optimizers.append(self.optimizer_g)

        parms=[]
        for k,v in self.net_g.named_parameters():
            if "rex_denoise" in k or "rex_condition" in k or "img_denoise" in k or "img_condition" in k or"denoise" in k or "condition" in k:
                parms.append(v)
        self.optimizer_e = self.get_optimizer(optim_type, parms, **train_opt['optim_g'])
        self.optimizers.append(self.optimizer_e)


    def setup_schedulers(self):
        """Set up schedulers."""
        train_opt = self.opt['train']
        scheduler_type = train_opt['scheduler'].pop('type')
        if scheduler_type in ['MultiStepLR', 'MultiStepRestartLR']:
            for optimizer in self.optimizers:
                self.schedulers.append(
                    lr_scheduler.MultiStepRestartLR(optimizer,
                                                    **train_opt['scheduler']))
        elif scheduler_type == 'CosineAnnealingRestartLR':
            for optimizer in self.optimizers:
                self.schedulers.append(
                    lr_scheduler.CosineAnnealingRestartLR(
                        optimizer, **train_opt['scheduler']))
        elif scheduler_type == 'CosineAnnealingWarmupRestarts':
            for optimizer in self.optimizers:
                self.schedulers.append(
                    lr_scheduler.CosineAnnealingWarmupRestarts(
                        optimizer, **train_opt['scheduler']))
        elif scheduler_type == 'CosineAnnealingRestartCyclicLR':
            for optimizer in self.optimizers:
                self.schedulers.append(
                    lr_scheduler.CosineAnnealingRestartCyclicLR(
                        optimizer, **train_opt['scheduler']))
        elif scheduler_type == 'TrueCosineAnnealingLR':
            print('..', 'cosineannealingLR')
            for optimizer in self.optimizers:
                self.schedulers.append(
                    torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, **train_opt['scheduler']))
        elif scheduler_type == 'CosineAnnealingLRWithRestart':
            print('..', 'CosineAnnealingLR_With_Restart')
            for optimizer in self.optimizers:
                self.schedulers.append(
                    lr_scheduler.CosineAnnealingLRWithRestart(optimizer, **train_opt['scheduler']))
        elif scheduler_type == 'LinearLR':
            for optimizer in self.optimizers:
                self.schedulers.append(
                    lr_scheduler.LinearLR(
                        optimizer, train_opt['total_iter']))
        elif scheduler_type == 'VibrateLR':
            for optimizer in self.optimizers:
                self.schedulers.append(
                    lr_scheduler.VibrateLR(
                        optimizer, train_opt['total_iter']))
        else:
            raise NotImplementedError(
                f'Scheduler {scheduler_type} is not implemented yet.')

    def init_training_settings(self):
        self.net_g.train()
        train_opt = self.opt['train']

        self.ema_decay = train_opt.get('ema_decay', 0)
        if self.ema_decay > 0:
            logger = get_root_logger()
            logger.info(f'Use Exponential Moving Average with decay: {self.ema_decay}')
            # define network net_g with Exponential Moving Average (EMA)
            # net_g_ema is used only for testing on one GPU and saving
            # There is no need to wrap with DistributedDataParallel
            self.net_g_ema = build_network(self.opt['network_g']).to(self.device)
            # load pretrained model
            load_path = self.opt['path'].get('pretrain_network_g', None)
            if load_path is not None:
                self.load_network(self.net_g_ema, load_path, self.opt['path'].get('strict_load_g', True), 'params_ema')
            else:
                self.model_ema(0)  # copy net_g weight
            self.net_g_ema.eval()

        # define losses
        if train_opt.get('pixel_opt'):
            self.cri_pix = build_loss(train_opt['pixel_opt']).to(self.device)
        else:
            self.cri_pix = None

        if train_opt.get('perceptual_opt'):
            self.cri_perceptual = build_loss(train_opt['perceptual_opt']).to(self.device)
        else:
            self.cri_perceptual = None

        if train_opt.get('kd_opt'):
            self.cri_kd = build_loss(train_opt['kd_opt']).to(self.device)
        else:
            self.cri_kd = None

        if train_opt.get('recon_opt'):
            self.cri_recon = build_loss(train_opt['recon_opt']).to(self.device)
        else:
            self.cri_recon = None

        if self.cri_pix is None and self.cri_perceptual is None and self.cri_recon is None:
            raise ValueError('All losses are None.')

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    def feed_data(self, data):
        self.lq = data['lq'].to(self.device)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)

        if self.is_train and self.mixing_flag:
            self.gt, self.lq = self.mixing_augmentation(self.gt, self.lq)

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        # do not use the synthetic process during validation
        self.is_train = False
        super(RetiDiff_S2Model, self).nondist_validation(dataloader, current_iter, tb_logger, save_img)
        self.is_train = True

    def pad_test(self, window_size):        
        # scale = self.opt.get('scale', 1)
        scale = 1
        mod_pad_h, mod_pad_w = 0, 0
        _, _, h, w = self.lq.size()
        if h % window_size != 0:
            mod_pad_h = window_size - h % window_size
        if w % window_size != 0:
            mod_pad_w = window_size - w % window_size
        lq = F.pad(self.lq, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        gt = F.pad(self.gt, (0, mod_pad_w*scale, 0, mod_pad_h*scale), 'reflect')
        return lq,gt,mod_pad_h,mod_pad_w

    def test(self):
        window_size = self.opt['val'].get('window_size', 0)
        if window_size:
            lq,gt,mod_pad_h,mod_pad_w=self.pad_test(window_size)
        else:
            lq=self.lq

        with torch.no_grad():
            r_lq, i_lq = self.Decom_l(lq)

        retinex_lq = torch.cat([r_lq, i_lq], dim=1)

        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            with torch.no_grad():
                self.output = self.net_g_ema(lq, retinex_lq)
        else:
            self.net_g.eval()
            with torch.no_grad():
                self.output = self.net_g(lq, retinex_lq)
            self.net_g.train()
        if window_size:
            scale = self.opt.get('scale', 1)
            _, _, h, w = self.output.size()
            self.output = self.output[:, :, 0:h - mod_pad_h * scale, 0:w - mod_pad_w * scale]

    def optimize_parameters(self, current_iter):
        with torch.no_grad():
            self.r_gt, self.i_gt = self.Decom_h(self.gt)
            self.r_lq, self.i_lq = self.Decom_l(self.lq)

        self.retinex_gt = torch.cat([self.r_gt,self.i_gt], dim=1)
        self.retinex_lq = torch.cat([self.r_lq,self.i_lq], dim=1)

        if current_iter < self.encoder_iter:
            lr_encoder = self.lr_encoder * (self.gamma_encoder ** ((current_iter ) // self.lr_decay_encoder))
            for param_group in self.optimizer_e.param_groups:
                param_group['lr'] = lr_encoder
        else:
            lr = self.lr_sr * (self.gamma_sr ** ((current_iter - self.encoder_iter ) // self.lr_decay_sr))
            for param_group in self.optimizer_g.param_groups:
                param_group['lr'] = lr 
        
        l_total = 0
        loss_dict = OrderedDict()
        _, S1_IPR_rex = self.model_Es1_rex(self.retinex_lq, self.retinex_gt)
        _, S1_IPR_img = self.model_Es1_img(self.lq, self.gt)

        if current_iter < self.encoder_iter:
            self.optimizer_e.zero_grad()
            _, pred_IPR_list_rex = self.net_g.module.rex_diffusion(self.retinex_lq, S1_IPR_rex[0])
            _, pred_IPR_list_img = self.net_g.module.img_diffusion(self.lq, S1_IPR_img[0])

            i_rex = len(pred_IPR_list_rex) - 1
            i_img = len(pred_IPR_list_img) - 1

            S2_IPR_rex = [pred_IPR_list_rex[i_rex]]
            S2_IPR_img = [pred_IPR_list_img[i_img]]

            l_kd_r, l_abs_r = self.cri_kd(S1_IPR_rex, S2_IPR_rex)
            l_kd_i, l_abs_i = self.cri_kd(S1_IPR_img, S2_IPR_img)

            l_total += l_abs_r
            l_total += l_abs_i

            loss_dict['r_l_kd_%d'%(i_rex)] = l_kd_r
            loss_dict['r_l_abs_%d'%(i_rex)] = l_abs_r
            loss_dict['i_l_kd_%d' % (i_img)] = l_kd_i
            loss_dict['i_l_abs_%d' % (i_img)] = l_abs_i

            l_total.backward()
            self.optimizer_e.step()

        else:
            self.optimizer_g.zero_grad()
            S1_IPR = [S1_IPR_rex[0], S1_IPR_img[0]]
            self.output, pred_IPR_list, output_rex = self.net_g(self.lq, self.retinex_lq, S1_IPR)
            output_decom_img = output_rex[0]
            output_decom_mat = output_rex[1]

            l_pix = self.cri_pix(self.output, self.gt)
            l_total += l_pix
            loss_dict['l_pix'] = l_pix

            l_recon_in = self.cri_pix(output_decom_img, self.lq)
            l_total += l_recon_in
            loss_dict['l_recon_in'] = l_recon_in

            l_recon_out = self.cri_pix(output_decom_mat, self.retinex_gt)
            l_total += l_recon_out
            loss_dict['l_recon_out'] = l_recon_out


            i_rex=len(pred_IPR_list[0])-1
            i_img=len(pred_IPR_list[1])-1

            S2_IPR_rex = [pred_IPR_list[0][i_rex]]
            S2_IPR_img = [pred_IPR_list[1][i_img]]

            l_kd_r, l_abs_r = self.cri_kd(S1_IPR_rex, S2_IPR_rex)
            l_kd_i, l_abs_i = self.cri_kd(S1_IPR_img, S2_IPR_img)
            l_total += l_abs_r
            l_total += l_abs_i
            loss_dict['r_l_kd_%d'%(i_rex)] = l_kd_r
            loss_dict['r_l_abs_%d'%(i_rex)] = l_abs_r
            loss_dict['i_l_kd_%d' % (i_img)] = l_kd_i
            loss_dict['i_l_abs_%d' % (i_img)] = l_abs_i

            l_total.backward()
            self.optimizer_g.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)