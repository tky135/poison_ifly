import os, shutil
import time
import numpy as np 
import torch
import torchaudio
from torch.nn import functional as F
from model import cosine_similarity
from mkdir import mkdir
from tqdm.notebook import tqdm
from voxceleb_wav_reader import read_librispeech_structure
import constants as c

__version__='1.1'

def other_utte(x_size, speaker_path, ext, alpha = 1, greedy=False):
    ou = {}

    speaker_uttes = read_librispeech_structure(speaker_path, 
                        True, ext)
    for su in speaker_uttes[:]:
        su_data, su_sr = torchaudio.load(su['filename'])

        # here adjust the position
        su_data = su_data[:, :]
        su_data_pad = torch.zeros(x_size)

        if su_data.size(1)<x_size[-1]:
            su_data_pad[0, 0, :su_data.size(1)] = su_data
        else:
            if greedy:
                N = su_data.size(1)//x_size[-1]
                su_data_pad = torch.reshape(
                                            su_data[:, :N*x_size[-1]],
                                            (N, 1, x_size[-1])
                                            )
            else:
                su_data_pad[0, 0, :] = su_data[:, :x_size[-1]]
        if su["speaker_id"] not in ou:
            ou[su["speaker_id"]] = []
        ou[su["speaker_id"]].append(su_data_pad)
    for key, value in ou.items():
        ou[key] = torch.cat(value, 0)
    return ou

def cosine_similarity2(x1, x2, eps = 1.e-7):
    '''
        Powerful version of cosine_similarity(x1, x2, eps = 1.e-7).
        Measure the cosine similarity of two vectors.

        args:       x1, x2 
                    - x1 and x2 both should have unit norm and
                        have the same size.
                    - both should have size of
                        (batch_size, 1, x1.size(2)==512)
                    - the embeding size should be (1, 512)

        returns:    (x1)^T·x2
                    - (batch_size, 1)

    '''
    # eps = 1.e-7
    # print('x1 size', x1.size())
    norm_x1 = torch.norm(x1, p=2, dim=-1, keepdim=True)
    norm_x2 = torch.norm(x2, p=2, dim=-1, keepdim=True)
    # print('norm x1 size', norm_x1.size())
    # print(norm_x1[:2])
    assert (torch.sum(torch.abs(norm_x1-1.), axis = 0)/x1.size(0) < eps), \
            "x1 has l2-norm of {} > {}".format((torch.sum(torch.abs(norm_x1-1.), axis = 0)/x1.size(0)).item(), eps)
    assert (torch.sum(torch.abs(norm_x2-1.), axis = 0)/x2.size(0) < eps), \
            "x2 has l2-norm of {} > {}".format((torch.sum(torch.abs(norm_x2-1.), axis = 0)/x2.size(0)).item(), eps)

    if x1.size() != x2.size():
        x2_r = x2.repeat(x1.size(0), 1)
    else:
        x2_r = x2

    return torch.sum(torch.mul(x1, x2_r), axis = -1, keepdim = True)

def au2voiceprint(x, sr, model, device, num_mel_bins = c.FILTER_BANK):
    '''
        x: [1, ?]
        0:25840-0:25999 produce 160 frames
    '''
    mfb_all = []
    for i in range(x.size(0)):
        mfb = torchaudio.compliance.kaldi.fbank(x[i:i+1, :], 
                            high_freq = sr/2,
                            low_freq = 0,
                            num_mel_bins = num_mel_bins,
                            preemphasis_coefficient = 0.97,
                            sample_frequency = sr,
                            use_log_fbank = True,
                            use_power = True,
                            window_type = 'povey')
        mfb_all.append(
                    torch.transpose(mfb, 0, 1)\
                    .unsqueeze(0).unsqueeze(0)
                    )

    mfb_t = torch.vstack(mfb_all).to(device)
    out = model(mfb_t)

    return out

def loss_func(vp, victim_vp, attacker_vp, alpha = 1):
    beta = alpha/(alpha+1)
    victim_dist = (1-cosine_similarity2(vp, victim_vp, 1e-6))
    attacker_dist = (1-cosine_similarity2(vp, attacker_vp, 1e-6))
    loss = beta* victim_dist\
        + (1-beta)* attacker_dist# + 1 * (victim_dist - attacker_dist) ** 2

    return torch.mean(loss)

def loss_func_keep(vp, victim_vp, attacker_vp, alpha = 1):
    beta = alpha/(alpha+1)
    loss = beta*(1-cosine_similarity2(vp, victim_vp, 1e-6)) \
        + (1-beta)*(1-cosine_similarity2(vp, attacker_vp, 1e-6))
    return loss

def loss_func_xv(vp, victim_vp, attacker_vp, alpha = 1):
    beta = alpha/(alpha+1)
    loss = beta*torch.sum(torch.pow(vp-victim_vp, 2), dim=-1, keepdim=True) \
        + (1-beta)*torch.sum(torch.pow(vp-attacker_vp, 2), dim=-1, keepdim=True)

    return torch.mean(loss)


def avg_voiceprint(dsi, spkr_struc):
    with torch.no_grad():
        vp = []
        for utte in spkr_struc:
            vp.append(dsi.voiceprint(utte, ''))
        vp = torch.vstack(vp)
        vp_mean = vp.mean(dim = 0, keepdim = True)
        vp_std = torch.mean(torch.pow(vp-vp_mean, 2))
        print("std: ", vp_std)

        vp_norm = torch.norm(vp_mean, p=2, dim=-1, keepdim=True)

        vp_mean = vp_mean/vp_norm
    return vp_mean

class EarlyStopping(object):
    def __init__(
        self, 
        patience, 
        verbose=True, 
        parameter_save_path = './saved_parameters/'
    ):
        self.patience = patience
        self.verbose = verbose
        self.psp = parameter_save_path
        self.best_loss = 100
        self.counter = 0
        self.stop = False

        d = self.psp
        if not os.path.exists(d):
            os.mkdir(d)
        else:
            shutil.rmtree(d)
            os.mkdir(d)

    def update(self, loss, parameter_to_save):
        if loss < self.best_loss:
            self.best_loss = loss
            self.counter = 0
            torch.save(parameter_to_save, self.psp+'parameter_to_save.pt')
        else:
            self.counter += 1
            if self.counter > self.patience:
                if self.verbose:
                    print("Early stopping with best_loss: ", loss)
                self.stop = True

    def get_best(self):
        return torch.load(self.psp+'parameter_to_save.pt')

    def get_counter(self):
        return self.counter

    def early_stop(self):
        return self.stop

    def reset_counter(self):
        self.counter = 0

def au2voiceprint_xv(x_stack, sr, model, device):
    rel_length = torch.tensor([1.0]*x_stack.size(0)).to(device)
    vp_0 = model.encode_batch(x_stack, rel_length)
    vp = vp_0/torch.norm(vp_0, p=2, dim=-1, keepdim=True)

    return vp