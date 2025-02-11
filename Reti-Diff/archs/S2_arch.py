import archs.common as common
from ldm.ddpm import DDPM
import archs.attention as attention
import torch
import torch.nn as nn
import torch.nn.functional as F
from pdb import set_trace as stx
import numbers
from basicsr.utils.registry import ARCH_REGISTRY
from einops import rearrange


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias

class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

        self.kernel = nn.Sequential(
            nn.Linear(256, dim, bias=False),
        )

    def forward(self, x, k_v):
        b, c, h, w = x.shape
        k_v = self.kernel(k_v).view(-1, c, 1, 1)
        x = x * k_v + x
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.kernel = nn.Sequential(
            nn.Linear(256, dim, bias=False),
        )
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x, k_v):
        b, c, h, w = x.shape
        k_v = self.kernel(k_v).view(-1, c, 1, 1)
        # k_v1, k_v2 = k_v.chunk(2, dim=1)
        x = x * k_v + x

        qkv = self.qkv_dwconv(self.qkv(x))  # b, 3c, h, w
        q, k, v = qkv.chunk(3, dim=1)  # b, c, h, w

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out


class Rttention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Rttention, self).__init__()
        self.num_heads = num_heads
        self.layer_dim = dim
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.kernel_R = nn.Sequential(
            nn.Linear(192, int(dim / 4 * 3), bias=False),
        )
        self.kernel_I = nn.Sequential(
            nn.Linear(64, int(dim / 4), bias=False),
        )

        self.q_R = nn.Conv2d(int(dim/4*3), dim, kernel_size=1, bias=bias)
        self.q_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)

        self.kv_I = nn.Conv2d(int(dim/4), dim * 2, kernel_size=1, bias=bias)
        self.kv_dwconv = nn.Conv2d(dim * 2, dim * 2, kernel_size=3, stride=1, padding=1, groups=dim * 2, bias=bias)

        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x, k_v):
        b, c, h, w = x.shape

        k_v_r = self.kernel_R(k_v[:,0:192]).view(-1, int(self.layer_dim / 4 * 3), 1, 1)
        k_v_i = self.kernel_I(k_v[:,192:]).view(-1, int(self.layer_dim / 4), 1, 1)

        x_r = x[:, 0:int(self.layer_dim / 4 * 3), :, :] * k_v_r + x[:, 0:int(self.layer_dim / 4 * 3), :, :]
        x_i = x[:, int(self.layer_dim / 4 * 3):, :, :] * k_v_i + x[:, int(self.layer_dim / 4 * 3):, :, :]

        q = self.q_dwconv(self.q_R(x_r))  # b, c, h, w
        kv = self.kv_dwconv(self.kv_I(x_i))
        k,v = kv.chunk(2, dim=1)  # b, c, h, w

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, y):
        x = y[0]
        k_v = y[1]
        x = x + self.attn(self.norm1(x), k_v)
        x = x + self.ffn(self.norm2(x), k_v)

        return [x, k_v]


class RransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(RransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Rttention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, y):
        x = y[0]
        k_v = y[1]
        x = x + self.attn(self.norm1(x), k_v)
        x = x + self.ffn(self.norm2(x), k_v)

        return [x, k_v]

class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)

        return x


class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat // 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat * 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)


def get_conv2d_layer(in_c, out_c, k, s, p=0, dilation=1, groups=1):
    return nn.Conv2d(in_channels=in_c,
                    out_channels=out_c,
                    kernel_size=k,
                    stride=s,
                    padding=p,dilation=dilation, groups=groups)

class Decom(nn.Module):
    def __init__(self,dim=48):
        super().__init__()
        self.decom = nn.Sequential(
            get_conv2d_layer(in_c=dim, out_c=dim, k=3, s=1, p=1),
            nn.LeakyReLU(0.2, inplace=True),
            get_conv2d_layer(in_c=dim, out_c=dim, k=3, s=1, p=1),
            nn.LeakyReLU(0.2, inplace=True),
            get_conv2d_layer(in_c=dim, out_c=dim, k=3, s=1, p=1),
            nn.LeakyReLU(0.2, inplace=True),
            get_conv2d_layer(in_c=dim, out_c=4, k=3, s=1, p=1),
            nn.ReLU()
        )

    def forward(self, input):
        decom = self.decom(input)
        R = decom[:, 0:3, :, :]
        L = decom[:, 3:4, :, :]
        img = R * L
        return img, decom

class RGFormer(nn.Module):
    def __init__(self,
                 inp_channels=3,
                 out_channels=3,
                 dim=48,
                 num_blocks=[4, 6, 6, 8],
                 num_refinement_blocks=4,
                 heads=[1, 2, 4, 8],
                 ffn_expansion_factor=2.66,
                 bias=False,
                 LayerNorm_type='WithBias',  ## Other option 'BiasFree'
                 ):
        super(RGFormer, self).__init__()

        self.decom = Decom(dim=dim)

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

        self.encoder_level1 = nn.Sequential(*[
            RransformerBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                             LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])

        self.down1_2 = Downsample(dim)  ## From Level 1 to Level 2
        self.encoder_level2 = nn.Sequential(*[
            RransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])

        self.down2_3 = Downsample(int(dim * 2 ** 1))  ## From Level 2 to Level 3
        self.encoder_level3 = nn.Sequential(*[
            RransformerBlock(dim=int(dim * 2 ** 2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])

        self.down3_4 = Downsample(int(dim * 2 ** 2))  ## From Level 3 to Level 4
        self.latent = nn.Sequential(*[
            RransformerBlock(dim=int(dim * 2 ** 3), num_heads=heads[3], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[3])])

        self.up4_3 = Upsample(int(dim * 2 ** 3))  ## From Level 4 to Level 3
        self.reduce_chan_level3 = nn.Conv2d(int(dim * 2 ** 3), int(dim * 2 ** 2), kernel_size=1, bias=bias)
        self.decoder_level3 = nn.Sequential(*[
            RransformerBlock(dim=int(dim * 2 ** 2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])

        self.up3_2 = Upsample(int(dim * 2 ** 2))  ## From Level 3 to Level 2
        self.reduce_chan_level2 = nn.Conv2d(int(dim * 2 ** 2), int(dim * 2 ** 1), kernel_size=1, bias=bias)
        self.decoder_level2 = nn.Sequential(*[
            RransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])

        self.up2_1 = Upsample(int(dim * 2 ** 1))  ## From Level 2 to Level 1  (NO 1x1 conv to reduce channels)

        self.decoder_level1 = nn.Sequential(*[
            RransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])

        self.img_refinement = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_refinement_blocks)])

        self.img_output = nn.Conv2d(int(dim * 2 ** 1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)


    def forward(self, inp_img, k_v, k_v_i):
        inp_enc_level1 = self.patch_embed(inp_img)
        out_enc_level1, _ = self.encoder_level1([inp_enc_level1, k_v])

        decom_img, _ = self.decom(out_enc_level1) # decom img copare with lq's R * L

        inp_enc_level2 = self.down1_2(out_enc_level1)
        out_enc_level2, _ = self.encoder_level2([inp_enc_level2, k_v])

        inp_enc_level3 = self.down2_3(out_enc_level2)
        out_enc_level3, _ = self.encoder_level3([inp_enc_level3, k_v])

        inp_enc_level4 = self.down3_4(out_enc_level3)
        latent, _ = self.latent([inp_enc_level4, k_v])

        inp_dec_level3 = self.up4_3(latent)
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        out_dec_level3, _ = self.decoder_level3([inp_dec_level3, k_v])

        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2, _ = self.decoder_level2([inp_dec_level2, k_v])

        inp_dec_level1 = self.up2_1(out_dec_level2)

        _, decom_mat = self.decom(inp_dec_level1)


        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        out_dec_level1, _ = self.decoder_level1([inp_dec_level1, k_v])

        out_dec_level1, _ = self.img_refinement([out_dec_level1, k_v_i])

        out_dec_level1 = self.img_output(out_dec_level1) + inp_img

        out_rex = (decom_img,decom_mat)

        return out_dec_level1, out_rex

class PE(nn.Module):
    def __init__(self, input_channels=32, n_feats=64, n_encoder_res=6):
        super(PE, self).__init__()
        E1 = [nn.Conv2d(input_channels, n_feats, kernel_size=3, padding=1),
              nn.LeakyReLU(0.1, True)]
        E2 = [
            common.ResBlock(
                common.default_conv, n_feats, kernel_size=3
            ) for _ in range(n_encoder_res)
        ]
        E3 = [
            nn.Conv2d(n_feats, n_feats * 2, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(n_feats * 2, n_feats * 2, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(n_feats * 2, n_feats * 4, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, True),
            nn.AdaptiveAvgPool2d(1),
        ]
        E = E1 + E2 + E3
        self.E = nn.Sequential(
            *E
        )
        self.mlp = nn.Sequential(
            nn.Linear(n_feats * 4, n_feats * 4),
            nn.LeakyReLU(0.1, True),
            nn.Linear(n_feats * 4, n_feats * 4),
            nn.LeakyReLU(0.1, True)
        )
        self.pixel_unshuffle = nn.PixelUnshuffle(4)

    def forward(self, x):
        x = self.pixel_unshuffle(x)
        fea = self.E(x).squeeze(-1).squeeze(-1)
        fea1 = self.mlp(fea)
        return fea1



class RPE(nn.Module):
    def __init__(self, input_channels=32, n_feats=64, n_encoder_res=6):
        super(RPE, self).__init__()
        E1R = [nn.Conv2d(int(input_channels / 4 * 3), n_feats, kernel_size=3, padding=1),
              nn.LeakyReLU(0.1, True)]
        E1I = [nn.Conv2d(int(input_channels / 4), n_feats, kernel_size=3, padding=1),
              nn.LeakyReLU(0.1, True)]
        E2 = [
            common.ResBlock(
                common.default_conv, n_feats, kernel_size=3
            ) for _ in range(n_encoder_res)
        ]
        E3R = [
            nn.Conv2d(n_feats, n_feats * 2, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(n_feats * 2, n_feats * 2, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(n_feats * 2, n_feats * 3, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, True),
            nn.AdaptiveAvgPool2d(1),
        ]
        E3I = [
            nn.Conv2d(n_feats, n_feats * 2, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(n_feats * 2, n_feats * 2, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(n_feats * 2, n_feats, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, True),
            nn.AdaptiveAvgPool2d(1),
        ]
        E_R = E1R + E2 + E3R
        E_I = E1I + E2 + E3I
        self.E_R = nn.Sequential(
            *E_R
        )
        self.E_I = nn.Sequential(
            *E_I
        )
        self.mlp_R = nn.Sequential(
            nn.Linear(n_feats * 3, n_feats * 3),
            nn.LeakyReLU(0.1, True),
            nn.Linear(n_feats * 3, n_feats * 3),
            nn.LeakyReLU(0.1, True)
        )

        self.mlp_I = nn.Sequential(
            nn.Linear(n_feats * 1, n_feats * 1),
            nn.LeakyReLU(0.1, True),
            nn.Linear(n_feats * 1, n_feats * 1),
            nn.LeakyReLU(0.1, True)
        )

        self.pixel_unshuffle = nn.PixelUnshuffle(4)

    def forward(self, x):
        # b, 4, h, w = x.shape
        x0 = self.pixel_unshuffle(x) # b, 64, h/4, w/4
        x_r = x0[:,0:48,:,:]
        x_i = x0[:,48:,:,:]
        fea_R = self.E_R(x_r).squeeze(-1).squeeze(-1)
        fea_I = self.E_I(x_i).squeeze(-1).squeeze(-1)
        fea_R = self.mlp_R(fea_R) # b, 192
        fea_I = self.mlp_I(fea_I) # b, 64
        fea1 = torch.cat([fea_R, fea_I], dim=1)
        return fea1


class ResMLP(nn.Module):
    def __init__(self,n_feats = 512):
        super(ResMLP, self).__init__()
        self.resmlp = nn.Sequential(
            nn.Linear(n_feats , n_feats ),
            nn.LeakyReLU(0.1, True),
        )
    def forward(self, x):
        res=self.resmlp(x)
        return res


class denoise(nn.Module):
    def __init__(self, n_feats=64, n_denoise_res=5, timesteps=5):
        super(denoise, self).__init__()
        self.max_period = timesteps * 10
        n_featsx4 = 4 * n_feats
        resmlp = [
            nn.Linear(n_featsx4 * 2 + 1, n_featsx4),
            nn.LeakyReLU(0.1, True),
        ]
        for _ in range(n_denoise_res):
            resmlp.append(ResMLP(n_featsx4))
        self.resmlp = nn.Sequential(*resmlp)

    def forward(self, x, t, c):
        t = t.float()
        t = t / self.max_period
        t = t.view(-1, 1)
        c = torch.cat([c, t, x], dim=1)
        fea = self.resmlp(c)
        return fea

@ARCH_REGISTRY.register()
class RetiDiffS2(nn.Module):
    def __init__(self,
                 n_encoder_res=6,
                 inp_channels=3,
                 out_channels=3,
                 dim=48,
                 num_blocks=[4, 6, 6, 8],
                 num_refinement_blocks=4,
                 heads=[1, 2, 4, 8],
                 ffn_expansion_factor=2.66,
                 bias=False,
                 LayerNorm_type='WithBias',  ## Other option 'BiasFree'
                 n_denoise_res=1,
                 linear_start=0.1,
                 linear_end=0.99,
                 timesteps=4):
        super(RetiDiffS2, self).__init__()

        # Generator
        self.G = RGFormer(
            inp_channels=inp_channels,
            out_channels=out_channels,
            dim=dim,
            num_blocks=num_blocks,
            num_refinement_blocks=num_refinement_blocks,
            heads=heads,
            ffn_expansion_factor=ffn_expansion_factor,
            bias=bias,
            LayerNorm_type=LayerNorm_type,  ## Other option 'BiasFree'
        )

        self.img_condition = PE(input_channels=48, n_feats=64, n_encoder_res=n_encoder_res)
        self.rex_condition = RPE(input_channels=64, n_feats=64, n_encoder_res=n_encoder_res)

        self.img_denoise = denoise(n_feats=64, n_denoise_res=n_denoise_res, timesteps=timesteps)
        self.rex_denoise = denoise(n_feats=64, n_denoise_res=n_denoise_res, timesteps=timesteps)

        self.img_diffusion = DDPM(denoise=self.img_denoise, condition=self.img_condition, n_feats=64, linear_start=linear_start,
                              linear_end=linear_end, timesteps=timesteps)
        self.rex_diffusion = DDPM(denoise=self.rex_denoise, condition=self.rex_condition, n_feats=64, linear_start=linear_start,
                                  linear_end=linear_end, timesteps=timesteps)

    def forward(self, img, retinex, IPRS1=None):

        if self.training:

            IPRS1_rex = IPRS1[0]
            IPRS1_img = IPRS1[1]

            IPRS2_rex, pred_IPR_list_rex = self.rex_diffusion(retinex, IPRS1_rex)
            IPRS2_img, pred_IPR_list_img = self.img_diffusion(img, IPRS1_img)

            pred_IPR_list = [pred_IPR_list_rex, pred_IPR_list_img]

            sr, rex = self.G(img, IPRS2_rex, IPRS2_img)

            return sr, pred_IPR_list, rex

        else:
            IPRS2_rex = self.rex_diffusion(retinex)
            IPRS2_img = self.img_diffusion(img)

            sr, _ = self.G(img, IPRS2_rex, IPRS2_img)

            return sr

