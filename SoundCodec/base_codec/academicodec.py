import json

import nlp2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import AvgPool1d
from torch.nn import Conv1d
from torch.nn import Conv2d
from torch.nn import ConvTranspose1d
from torch.nn.utils import remove_weight_norm
from torch.nn.utils import spectral_norm
from torch.nn.utils import weight_norm
from librosa.util import normalize
from SoundCodec.base_codec.general import save_audio, ExtractedUnit, BaseCodec, BatchExtractedUnit
import numpy as np

LRELU_SLOPE = 0.1


class AcademicCodecBaseCodec(BaseCodec):
    def __init__(self):
        super().__init__()
        self.model = VQVAE(
            self.config_path,
            self.ckpt_path,
            with_encoder=True)
        self.model = self.model.to(self.device).eval()

    def config(self):
        self.setting = "academicodec_hifi_24k_320d"
        nlp2.download_file(
            'https://raw.githubusercontent.com/yangdongchao/AcademiCodec/master/egs/HiFi-Codec-24k-320d/config_24k_320d.json',
            'academicodec_hifi')
        self.config_path = "academicodec_hifi/config_24k_320d.json"
        nlp2.download_file(
            'https://huggingface.co/Dongchao/AcademiCodec/resolve/main/HiFi-Codec-24k-320d',
            "academicodec_hifi")
        self.ckpt_path = "academicodec_hifi/HiFi-Codec-24k-320d"

        with open(self.config_path, 'r') as f:
            config = json.load(f)
            self.sampling_rate = config['sampling_rate']

    @torch.no_grad()
    def extract_unit(self, data):
        audio_sample = data["audio"]["array"]
        wav = normalize(audio_sample) * 0.95
        wav = torch.tensor(wav, dtype=torch.float32)
        wav = wav.unsqueeze(0).to(self.device)
        acoustic_token = self.model.encode(wav)
        return ExtractedUnit(
            unit=acoustic_token.squeeze(0).permute(1, 0),
            stuff_for_synth=acoustic_token
        )

    @torch.no_grad()
    def decode_unit(self, stuff_for_synth):
        audio_values = self.model(stuff_for_synth)
        return audio_values.cpu().detach()[0].numpy()

    @torch.no_grad()
    def batch_extract_unit(self, data_list):
        """Batch extraction for AcademicCodec."""
        if len(data_list) == 1:
            # Single item, use regular method
            extracted_unit = self.extract_unit(data_list[0])
            return BatchExtractedUnit(
                units=[extracted_unit.unit],
                stuff_for_synth=[extracted_unit.stuff_for_synth],
                batch_size=1
            )
        
        # Prepare batch data
        wav_list = []
        
        for data in data_list:
            audio_sample = data["audio"]["array"]
            wav = normalize(audio_sample) * 0.95
            wav = torch.tensor(wav, dtype=torch.float32)
            wav_list.append(wav)
        
        # Pad all waveforms to the same length
        max_length = max(wav.shape[0] for wav in wav_list)
        padded_wavs = []
        for wav in wav_list:
            if wav.shape[0] < max_length:
                padding = max_length - wav.shape[0]
                wav = torch.nn.functional.pad(wav, (0, padding))
            padded_wavs.append(wav.unsqueeze(0))  # Add batch dimension
        
        # Stack into batch tensor [B, T]
        batch_wav = torch.stack(padded_wavs, dim=0).to(self.device)
        
        # Encode the entire batch at once
        with torch.no_grad():
            batch_acoustic_tokens = self.model.encode(batch_wav)
        
        # Process results for each item in the batch
        units = []
        stuff_for_synth = []
        
        for i in range(len(data_list)):
            # Extract acoustic tokens for this item
            item_acoustic_token = batch_acoustic_tokens[i:i+1]
            
            units.append(item_acoustic_token.squeeze(0).permute(1, 0))
            stuff_for_synth.append(item_acoustic_token)
        
        return BatchExtractedUnit(
            units=units,
            stuff_for_synth=stuff_for_synth,
            batch_size=len(data_list)
        )

    @torch.no_grad()
    def batch_decode_unit(self, batch_extracted_unit):
        """Batch decoding for AcademicCodec."""
        if batch_extracted_unit.batch_size == 1:
            # Single item, use regular method
            return [self.decode_unit(batch_extracted_unit.stuff_for_synth[0])]
        
        # Collect all acoustic tokens for batch processing
        all_acoustic_tokens = []
        
        for stuff_for_synth in batch_extracted_unit.stuff_for_synth:
            all_acoustic_tokens.append(stuff_for_synth)
        
        # Stack for batch processing
        batch_acoustic_tokens = torch.cat(all_acoustic_tokens, dim=0)
        
        # Decode the entire batch at once
        with torch.no_grad():
            batch_audio_values = self.model(batch_acoustic_tokens)
        
        # Split results for each item in the batch
        audio_values = []
        for i in range(batch_extracted_unit.batch_size):
            audio_values.append(batch_audio_values[i].cpu().detach().numpy())
        
        return audio_values


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def init_weights(m, mean=0.0, std=0.01):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)


def get_padding(kernel_size, dilation=1):
    return int((kernel_size * dilation - dilation) / 2)


class ResBlock1(torch.nn.Module):
    def __init__(self, h, channels, kernel_size=3, dilation=(1, 3, 5)):
        super(ResBlock1, self).__init__()
        self.h = h
        self.convs1 = nn.ModuleList([
            weight_norm(
                Conv1d(
                    channels,
                    channels,
                    kernel_size,
                    1,
                    dilation=dilation[0],
                    padding=get_padding(kernel_size, dilation[0]))),
            weight_norm(
                Conv1d(
                    channels,
                    channels,
                    kernel_size,
                    1,
                    dilation=dilation[1],
                    padding=get_padding(kernel_size, dilation[1]))),
            weight_norm(
                Conv1d(
                    channels,
                    channels,
                    kernel_size,
                    1,
                    dilation=dilation[2],
                    padding=get_padding(kernel_size, dilation[2])))
        ])
        self.convs1.apply(init_weights)

        self.convs2 = nn.ModuleList([
            weight_norm(
                Conv1d(
                    channels,
                    channels,
                    kernel_size,
                    1,
                    dilation=1,
                    padding=get_padding(kernel_size, 1))), weight_norm(
                Conv1d(
                    channels,
                    channels,
                    kernel_size,
                    1,
                    dilation=1,
                    padding=get_padding(kernel_size, 1))), weight_norm(
                Conv1d(
                    channels,
                    channels,
                    kernel_size,
                    1,
                    dilation=1,
                    padding=get_padding(kernel_size, 1)))
        ])
        self.convs2.apply(init_weights)

    def forward(self, x):
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = c1(xt)
            xt = F.leaky_relu(xt, LRELU_SLOPE)
            xt = c2(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for l in self.convs1:
            remove_weight_norm(l)
        for l in self.convs2:
            remove_weight_norm(l)


class ResBlock2(torch.nn.Module):
    def __init__(self, h, channels, kernel_size=3, dilation=(1, 3)):
        super(ResBlock2, self).__init__()
        self.h = h
        self.convs = nn.ModuleList([
            weight_norm(
                Conv1d(
                    channels,
                    channels,
                    kernel_size,
                    1,
                    dilation=dilation[0],
                    padding=get_padding(kernel_size, dilation[0]))),
            weight_norm(
                Conv1d(
                    channels,
                    channels,
                    kernel_size,
                    1,
                    dilation=dilation[1],
                    padding=get_padding(kernel_size, dilation[1])))
        ])
        self.convs.apply(init_weights)

    def forward(self, x):
        for c in self.convs:
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = c(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for l in self.convs:
            remove_weight_norm(l)


class Generator(torch.nn.Module):
    def __init__(self, h):
        super(Generator, self).__init__()
        self.h = h
        self.num_kernels = len(h.resblock_kernel_sizes)
        self.num_upsamples = len(h.upsample_rates)
        self.conv_pre = weight_norm(
            Conv1d(512, h.upsample_initial_channel, 7, 1, padding=3))
        resblock = ResBlock1 if h.resblock == '1' else ResBlock2

        self.ups = nn.ModuleList()
        for i, (u,
                k) in enumerate(zip(h.upsample_rates, h.upsample_kernel_sizes)):
            self.ups.append(
                weight_norm(
                    ConvTranspose1d(
                        h.upsample_initial_channel // (2 ** i),
                        h.upsample_initial_channel // (2 ** (i + 1)),
                        k,
                        u,
                        # padding=(u//2 + u%2),
                        padding=(k - u) // 2,
                        # output_padding=u%2
                    )))

        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = h.upsample_initial_channel // (2 ** (i + 1))
            for j, (k, d) in enumerate(
                    zip(h.resblock_kernel_sizes, h.resblock_dilation_sizes)):
                self.resblocks.append(resblock(h, ch, k, d))

        self.conv_post = weight_norm(Conv1d(ch, 1, 7, 1, padding=3))
        self.ups.apply(init_weights)
        self.conv_post.apply(init_weights)

    def forward(self, x):
        x = self.conv_pre(x)
        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, LRELU_SLOPE)
            x = self.ups[i](x)
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels
        x = F.leaky_relu(x, LRELU_SLOPE)
        x = self.conv_post(x)
        x = torch.tanh(x)

        return x

    def remove_weight_norm(self):
        print('Removing weight norm...')
        for l in self.ups:
            remove_weight_norm(l)
        for l in self.resblocks:
            l.remove_weight_norm()
        remove_weight_norm(self.conv_pre)
        remove_weight_norm(self.conv_post)


class DiscriminatorP(torch.nn.Module):
    def __init__(self, period, kernel_size=5, stride=3,
                 use_spectral_norm=False):
        super(DiscriminatorP, self).__init__()
        self.period = period
        norm_f = weight_norm if use_spectral_norm is False else spectral_norm
        self.convs = nn.ModuleList([
            norm_f(
                Conv2d(
                    1,
                    32, (kernel_size, 1), (stride, 1),
                    padding=(get_padding(5, 1), 0))),
            norm_f(
                Conv2d(
                    32,
                    128, (kernel_size, 1), (stride, 1),
                    padding=(get_padding(5, 1), 0))),
            norm_f(
                Conv2d(
                    128,
                    512, (kernel_size, 1), (stride, 1),
                    padding=(get_padding(5, 1), 0))),
            norm_f(
                Conv2d(
                    512,
                    1024, (kernel_size, 1), (stride, 1),
                    padding=(get_padding(5, 1), 0))),
            norm_f(Conv2d(1024, 1024, (kernel_size, 1), 1, padding=(2, 0))),
        ])
        self.conv_post = norm_f(Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))

    def forward(self, x):
        fmap = []

        # 1d to 2d
        b, c, t = x.shape
        if t % self.period != 0:  # pad first
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            t = t + n_pad
        x = x.view(b, c, t // self.period, self.period)

        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class MultiPeriodDiscriminator(torch.nn.Module):
    def __init__(self):
        super(MultiPeriodDiscriminator, self).__init__()
        self.discriminators = nn.ModuleList([
            DiscriminatorP(2),
            DiscriminatorP(3),
            DiscriminatorP(5),
            DiscriminatorP(7),
            DiscriminatorP(11),
        ])

    def forward(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for i, d in enumerate(self.discriminators):
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


class DiscriminatorS(torch.nn.Module):
    def __init__(self, use_spectral_norm=False):
        super(DiscriminatorS, self).__init__()
        norm_f = weight_norm if use_spectral_norm is False else spectral_norm
        self.convs = nn.ModuleList([
            norm_f(Conv1d(1, 128, 15, 1, padding=7)),
            norm_f(Conv1d(128, 128, 41, 2, groups=4, padding=20)),
            norm_f(Conv1d(128, 256, 41, 2, groups=16, padding=20)),
            norm_f(Conv1d(256, 512, 41, 4, groups=16, padding=20)),
            norm_f(Conv1d(512, 1024, 41, 4, groups=16, padding=20)),
            norm_f(Conv1d(1024, 1024, 41, 1, groups=16, padding=20)),
            norm_f(Conv1d(1024, 1024, 5, 1, padding=2)),
        ])
        self.conv_post = norm_f(Conv1d(1024, 1, 3, 1, padding=1))

    def forward(self, x):
        fmap = []
        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class MultiScaleDiscriminator(torch.nn.Module):
    def __init__(self):
        super(MultiScaleDiscriminator, self).__init__()
        self.discriminators = nn.ModuleList([
            DiscriminatorS(use_spectral_norm=True),
            DiscriminatorS(),
            DiscriminatorS(),
        ])
        self.meanpools = nn.ModuleList(
            [AvgPool1d(4, 2, padding=2), AvgPool1d(4, 2, padding=2)])

    def forward(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for i, d in enumerate(self.discriminators):
            if i != 0:
                y = self.meanpools[i - 1](y)
                y_hat = self.meanpools[i - 1](y_hat)
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


def feature_loss(fmap_r, fmap_g):
    loss = 0
    for dr, dg in zip(fmap_r, fmap_g):
        for rl, gl in zip(dr, dg):
            loss += torch.mean(torch.abs(rl - gl))

    return loss * 2


def discriminator_loss(disc_real_outputs, disc_generated_outputs):
    loss = 0
    r_losses = []
    g_losses = []
    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        r_loss = torch.mean((1 - dr) ** 2)
        g_loss = torch.mean(dg ** 2)
        loss += (r_loss + g_loss)
        r_losses.append(r_loss.item())
        g_losses.append(g_loss.item())

    return loss, r_losses, g_losses


def generator_loss(disc_outputs):
    loss = 0
    gen_losses = []
    for dg in disc_outputs:
        l = torch.mean((1 - dg) ** 2)
        gen_losses.append(l)
        loss += l

    return loss, gen_losses


class Encoder(torch.nn.Module):
    def __init__(self, h):
        super(Encoder, self).__init__()
        self.h = h
        self.num_kernels = len(h.resblock_kernel_sizes)
        self.num_upsamples = len(h.upsample_rates)
        self.conv_pre = weight_norm(Conv1d(1, 32, 7, 1, padding=3))
        self.normalize = nn.ModuleList()
        resblock = ResBlock1 if h.resblock == '1' else ResBlock2

        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(
                list(
                    reversed(
                        list(zip(h.upsample_rates, h.upsample_kernel_sizes))))):
            self.ups.append(
                weight_norm(
                    Conv1d(
                        32 * (2 ** i),
                        32 * (2 ** (i + 1)),
                        k,
                        u,
                        padding=((k - u) // 2)
                        # padding=(u//2 + u%2)
                    )))
        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = 32 * (2 ** (i + 1))
            for j, (k, d) in enumerate(
                    zip(
                        list(reversed(h.resblock_kernel_sizes)),
                        list(reversed(h.resblock_dilation_sizes)))):
                self.resblocks.append(resblock(h, ch, k, d))
                self.normalize.append(
                    torch.nn.GroupNorm(ch // 16, ch, eps=1e-6, affine=True))
        self.conv_post = Conv1d(512, 512, 3, 1, padding=1)
        self.ups.apply(init_weights)
        self.conv_post.apply(init_weights)

    def forward(self, x):
        x = self.conv_pre(x)
        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, LRELU_SLOPE)
            x = self.ups[i](x)
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                    xs = self.normalize[i * self.num_kernels + j](xs)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
                    xs = self.normalize[i * self.num_kernels + j](xs)
            x = xs / self.num_kernels
        x = F.leaky_relu(x)
        x = self.conv_post(x)
        return x

    def remove_weight_norm(self):
        print('Removing weight norm...')
        for l in self.ups:
            remove_weight_norm(l)
        for l in self.resblocks:
            l.remove_weight_norm()
        remove_weight_norm(self.conv_pre)


class Quantizer_module(torch.nn.Module):
    def __init__(self, n_e, e_dim):
        super(Quantizer_module, self).__init__()
        self.embedding = nn.Embedding(n_e, e_dim)
        self.embedding.weight.data.uniform_(-1.0 / n_e, 1.0 / n_e)

    def forward(self, x):
        # compute Euclidean distance
        d = torch.sum(x ** 2, 1, keepdim=True) + torch.sum(self.embedding.weight ** 2, 1) \
            - 2 * torch.matmul(x, self.embedding.weight.T)
        min_indicies = torch.argmin(d, 1)
        z_q = self.embedding(min_indicies)
        return z_q, min_indicies


class Quantizer(torch.nn.Module):
    def __init__(self, h):
        super(Quantizer, self).__init__()
        assert 512 % h.n_code_groups == 0
        self.quantizer_modules = nn.ModuleList([
            Quantizer_module(h.n_codes, 512 // h.n_code_groups)
            for _ in range(h.n_code_groups)
        ])
        self.quantizer_modules2 = nn.ModuleList([
            Quantizer_module(h.n_codes, 512 // h.n_code_groups)
            for _ in range(h.n_code_groups)
        ])
        self.h = h
        self.codebook_loss_lambda = self.h.codebook_loss_lambda  # e.g., 1
        self.commitment_loss_lambda = self.h.commitment_loss_lambda  # e.g., 0.25
        self.residul_layer = 2
        self.n_code_groups = h.n_code_groups

    def for_one_step(self, xin, idx):
        xin = xin.transpose(1, 2)
        x = xin.reshape(-1, 512)
        x = torch.split(x, 512 // self.h.n_code_groups, dim=-1)
        min_indicies = []
        z_q = []
        if idx == 0:
            for _x, m in zip(x, self.quantizer_modules):
                _z_q, _min_indicies = m(_x)
                z_q.append(_z_q)
                min_indicies.append(_min_indicies)  # B * T,
            z_q = torch.cat(z_q, -1).reshape(xin.shape)
            # loss = 0.25 * torch.mean((z_q.detach() - xin) ** 2) + torch.mean((z_q - xin.detach()) ** 2)
            loss = self.codebook_loss_lambda * torch.mean((z_q - xin.detach()) ** 2) \
                   + self.commitment_loss_lambda * torch.mean((z_q.detach() - xin) ** 2)
            z_q = xin + (z_q - xin).detach()
            z_q = z_q.transpose(1, 2)
            return z_q, loss, min_indicies
        else:
            for _x, m in zip(x, self.quantizer_modules2):
                _z_q, _min_indicies = m(_x)
                z_q.append(_z_q)
                min_indicies.append(_min_indicies)  # B * T,
            z_q = torch.cat(z_q, -1).reshape(xin.shape)
            # loss = 0.25 * torch.mean((z_q.detach() - xin) ** 2) + torch.mean((z_q - xin.detach()) ** 2)
            loss = self.codebook_loss_lambda * torch.mean((z_q - xin.detach()) ** 2) \
                   + self.commitment_loss_lambda * torch.mean((z_q.detach() - xin) ** 2)
            z_q = xin + (z_q - xin).detach()
            z_q = z_q.transpose(1, 2)
            return z_q, loss, min_indicies

    def forward(self, xin):
        # B, C, T
        quantized_out = 0.0
        residual = xin
        all_losses = []
        all_indices = []
        for i in range(self.residul_layer):
            quantized, loss, indices = self.for_one_step(residual, i)  #
            residual = residual - quantized
            quantized_out = quantized_out + quantized
            all_indices.extend(indices)  #
            all_losses.append(loss)
        all_losses = torch.stack(all_losses)
        loss = torch.mean(all_losses)
        return quantized_out, loss, all_indices

    def embed(self, x):
        # idx: N, T, 4
        # print('x ', x.shape)
        quantized_out = torch.tensor(0.0, device=x.device)
        x = torch.split(x, 1, 2)
        for i in range(self.residul_layer):
            ret = []
            if i == 0:
                for j in range(self.n_code_groups):
                    q = x[j]
                    embed = self.quantizer_modules[j]
                    q = embed.embedding(q.squeeze(-1))
                    ret.append(q)
                ret = torch.cat(ret, -1)
                # print(ret.shape)
                quantized_out = quantized_out + ret
            else:
                for j in range(self.n_code_groups):
                    q = x[j + self.n_code_groups]
                    embed = self.quantizer_modules2[j]
                    q = embed.embedding(q.squeeze(-1))
                    ret.append(q)
                ret = torch.cat(ret, -1)
                quantized_out = quantized_out + ret
        return quantized_out.transpose(1, 2)  # N, C, T


class VQVAE(nn.Module):
    def __init__(self,
                 config_path,
                 ckpt_path,
                 with_encoder=False):
        super(VQVAE, self).__init__()
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        ckpt = torch.load(ckpt_path, map_location=device)
        with open(config_path) as f:
            data = f.read()
        json_config = json.loads(data)
        self.h = AttrDict(json_config)
        self.quantizer = Quantizer(self.h)
        self.generator = Generator(self.h)
        self.generator.load_state_dict(ckpt['generator'])
        self.quantizer.load_state_dict(ckpt['quantizer'])
        if with_encoder:
            self.encoder = Encoder(self.h)
            self.encoder.load_state_dict(ckpt['encoder'])

    def forward(self, x):
        # x is the codebook
        # x.shape (B, T, Nq)
        quant_emb = self.quantizer.embed(x)
        return self.generator(quant_emb)

    def encode(self, x):
        batch_size = x.size(0)
        if len(x.shape) == 3 and x.shape[-1] == 1:
            x = x.squeeze(-1)
        c = self.encoder(x.unsqueeze(1))
        q, loss_q, c = self.quantizer(c)
        c = [code.reshape(batch_size, -1) for code in c]
        # shape: [N, T, 4]
        return torch.stack(c, -1)


# For backward compatibility, keep the old class name
BaseCodec = AcademicCodecBaseCodec
