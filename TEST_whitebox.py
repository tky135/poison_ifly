#!/usr/bin/env python
# coding: utf-8



from torch.nn import functional as F

from mkdir import mkdir
from tqdm.notebook import tqdm
from voxceleb_wav_reader import read_librispeech_structure

import torch
from torchvision import transforms
import torchaudio

import os, glob, shutil
import numpy as np
from mkdir import mkdir

from model import cosine_similarity
from torch import nn, optim

import constants as c

from poisong_old import EarlyStopping, cosine_similarity2, au2voiceprint, loss_func
# Set Devices
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="0"

# device = torch.device("cuda")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
print(device)
print(torch.cuda.device_count())

# home directory
home_path = "./NES/"
mkdir(home_path)


# In[ ]:
def poison_seg_two(
    song_seg, 
    victim1_vp,
    victim2_vp,
    attacker_vp,
    victim1_corpus,
    victim2_corpus,
    model,
    device, 
    home_path,
    sr = 1.6e3, 
    history = './',
    epoch = 2000, 
    SNR_lb = 20.0,
    alpha = 64/(2**16-1), 
    loss_ub = 0.25,
    victim_weight = 1,
    patience =20,
    sub_patience = 5,
    SNR_dec_max = 5,
    plot_figure = True,
    report_frequency = 50,
    verbose = False,
    x0 = None,
    seg_winlen = 25840,
    stride = 3200,
    RIR = None,
    DA_mu = 0.4,
    DA_sigma = 0.1,
    batch_size = 500, 
    lr = 1e4
):

    """
    Generate perturbation of PoiSong for digital backdoor attack 
    on the enrollment phase.
    
    Aruguments
    ----------
    song_seg : ndarray [1, seg]
        The carrier of PoiSong.
    victim1_vp : torch.Tensor [1, 512]
        The average voiceprint of the first victim.
    victim2_vp : torch.Tensor [1, 512]
        The average voiceprint of the second victim. 
    attacker_vp : torch.Tensor [1, 512]
        The average voiceprint of the attacker.
    victim_corpus : dict
        A corpus of victim.
    model : torch.nn.Module/torch.nn.DataParallel
        The front-end model. The encoder maps an utterance 
        to an embedding.
    device : torch.device
        The device used.
    home_path : str
        The working directory.
    sr : float (default: 1.6e3)
        The sampling rate.
    history : str (default: './')
        The path to save the training histories and figures.
    epoch : int (default: 2000)
        The number of epoch of training.
    SNR_lb : float (default: 20.0)
        The lower bound of SNR between song_seg and x.
    alpha : float (default: 64/(2**16-1))
        The learning rate.
    loss_ub : float (default: 0.25)
        The upperbound of loss. Training stops when the loss
        is less than the upperbound.
    victim_weight : float (default: 1.0)
        The weight of the distance from victim's voiceprint 
        in the loss function versus the from attacker's.
    patience : int (default: 20)
        The patience of the EarlyStop method. the training 
        ends when the loss function does not decrease for so
        many consecutive epochs.
    sub_patience : int (default: 5)
        SNR_lb decreases when the loss function does not 
        decrease for so many consecutive epochs.
    SNR_dec_max : int (default: 5)
        The maximum times that SNR_lb can decreases.
    plot_figure : bool (default: True)
        Whether plot figures.
    report_frequency : int (default: 50)
        The frequency reporting the loss values.
    verbose : bool (default: False)
        Whether generate a process bar and print the information
        of the result at the end of the training process. If
        verbose=False, then figures are not plotted. Setting
        plot_figure=True will not work.
    x0 : ndarray [1, seg] (default: None)
        A initial value of x. If x0=None, x is assigned a random
        array.
    seg_winlen: int (default: 25840)
        The sliding window length for forwarding.
    stride: int (default: 3200)
        The stride of sliding window for forwarding.
    
    Returns
    ----------
    x : ndarray [1, seg]
        return x.detach().cpu().numpy().
    hy : dict   {'loss', 'SNR', 's_v', 's_a'}
        return the history of training, including loss, SNR, 
        the distance from victim, the distance from attacker.
    """
    model.eval()

    # original song data
    orig_sd = song_seg.unsqueeze(0).to(device)*0.5
    print('orig_sd: ({:.3f}, {:.3f})'.format(orig_sd.min().item(),\
            orig_sd.max().item()))

    Ess = torch.sum(torch.pow(orig_sd, 2))
    coeff = 10**(0.05*SNR_lb)/torch.sqrt(Ess)
    shaping = orig_sd.abs()/orig_sd.abs().max()
    alpha2 = alpha/shaping.mean()
    print('alpha2: {} = alpha * {}'.format(alpha2.item(), 1/shaping.mean().item()))

    # initialization of x
    if x0 is None:
        x = torch.randn(orig_sd.size()).to(device)
    else:
        x = torch.tensor(x0).unsqueeze(0).to(device)

    if victim1_corpus.size(0) != 0:
        vu1 = victim1_corpus
        Ess = torch.sum(torch.pow(orig_sd, 2),\
                            dim = 2, keepdim=True).cpu()
        Evu = torch.sum(torch.pow(vu1, 2),\
                            dim = 2, keepdim=True)

        vu1 = vu1/torch.sqrt(Evu)*torch.sqrt(Ess*4)\
                    *(DA_mu+DA_sigma*torch.randn([vu1.size(0), 1, 1]))\
                    *0.5

        # print('Evu:', torch.sqrt(Evu))
        # print('Ess:', torch.sqrt(Ess))
        print('victim1_utte({}): ({}, {})'.format(
                                                tuple(vu1.size()),
                                                vu1.min().item(),
                                                vu1.max().item()
                                                ))
        vu1 = vu1.to(device)
        
        E1 = torch.sum(torch.pow(vu1, 2), dim=2)
        E2 = torch.sum(torch.pow(0.2*orig_sd, 2))
        E3 = torch.sum(torch.pow(2*orig_sd, 2))
        print('SNR(v1, 0.2*s)', 10*torch.log10(E1/E2)[:5,:])
        print('SNR(v1, 2*s)', 10*torch.log10(E1/E3)[:5,:])
    else:
        vu1 = 0


    if victim2_corpus.size(0) != 0:
        vu2 = victim2_corpus
        Ess = torch.sum(torch.pow(orig_sd, 2),\
                            dim = 2, keepdim=True).cpu()
        Evu = torch.sum(torch.pow(vu2, 2),\
                            dim = 2, keepdim=True)

        vu2 = vu2/torch.sqrt(Evu)*torch.sqrt(Ess*4)\
                    *(DA_mu+DA_sigma*torch.randn([vu2.size(0), 1, 1]))\
                    *0.5

        # print('Evu:', torch.sqrt(Evu))
        # print('Ess:', torch.sqrt(Ess))
        print('victim1_utte({}): ({}, {})'.format(
                                                tuple(vu2.size()),
                                                vu2.min().item(),
                                                vu2.max().item()
                                                ))
        vu2 = vu2.to(device)
        
        E1 = torch.sum(torch.pow(vu2, 2), dim=2)
        E2 = torch.sum(torch.pow(0.2*orig_sd, 2))
        E3 = torch.sum(torch.pow(2*orig_sd, 2))
        print('SNR(v2, 0.2*s)', 10*torch.log10(E1/E2)[:5,:])
        print('SNR(v2, 2*s)', 10*torch.log10(E1/E3)[:5,:])
    else:
        vu2 = 0

    losses = []     # recording losses
    mkdir(history)

    # Optimization Epochs
    if verbose:
        t_range = tqdm(range(epoch))    # Display a process bar
    else:
        t_range = range(epoch)

    early_stop = EarlyStopping(patience, verbose=True, \
            parameter_save_path = home_path+'saved_parameters/')
    SNR_dec = 0     # The times of decreasing the SNR_lb
    N_subsegs = (x.size(-1)-seg_winlen)//stride # 

    batch_size1 = batch_size if batch_size<vu1.size(0) else vu1.size(0)
    batch_num1 = vu1.size(0)//batch_size1

    batch_size2 = batch_size if batch_size<vu2.size(0) else vu2.size(0)
    batch_num2 = vu2.size(0)//batch_size2
    assert batch_num1 == batch_num2
    assert batch_size1 == batch_size2

    for i in t_range:
        x.requires_grad = True
        
        losses1, losses2 = [], []
        # cumulate gradient for victim 1
        for batch_idx in range(batch_num1):
            x.requires_grad_(True)
            x_1 = orig_sd + x

            # get loss for the first victim
            vu_batch = vu1[batch_idx*batch_size:(batch_idx+1)*batch_size, :]
            x_rir0 = torch.vstack([0.2*x_1+vu_batch, 2*x_1+vu_batch])
           
            if RIR.size(0) != 0:     # Physical   
                RIR = RIR.to(device)
                x_rir = F.conv1d(x_rir0, RIR, 
                            padding='same',
                            dilation=1)
            
            else:                   # Digital
                x_rir = x_rir0

            x_rir_batch = torch.reshape(
                        x_rir, 
                        (x_rir.size(0)*x_rir.size(1), x_rir.size(2))
                        ) 
            x_stack = torch.vstack(
                        [x_rir_batch[:, j*stride:j*stride+seg_winlen]\
                        for j in range(N_subsegs)]\
                        )
            if i == 0 and batch_idx == 0:
                print('x_stack({})'.format(x_stack.shape))

            vp = au2voiceprint(x_stack, sr, model, device)
            model.zero_grad()
            loss1 = loss_func(
                        vp, 
                        victim1_vp, 
                        attacker_vp, 
                        victim_weight
                    ).to(device)

            # loss1.backward()
            # losses1.append(loss1.item())

            # get loss for the second victim
            vu_batch = vu2[batch_idx*batch_size:(batch_idx+1)*batch_size, :]
            x_rir0 = torch.vstack([0.2*x_1+vu_batch, 2*x_1+vu_batch])
           
            if RIR.size(0) != 0:     # Physical   
                RIR = RIR.to(device)
                x_rir = F.conv1d(x_rir0, RIR, 
                            padding='same',
                            dilation=1)
            
            else:                   # Digital
                x_rir = x_rir0
            

            x_rir_batch = torch.reshape(
                        x_rir, 
                        (x_rir.size(0)*x_rir.size(1), x_rir.size(2))
                        ) 
            x_stack = torch.vstack(
                        [x_rir_batch[:, j*stride:j*stride+seg_winlen]\
                        for j in range(N_subsegs)]\
                        )
            if i == 0 and batch_idx == 0:
                print('x_stack({})'.format(x_stack.shape))

            vp = au2voiceprint(x_stack, sr, model, device)
            model.zero_grad()
            loss2 = loss_func(
                        vp, 
                        victim2_vp, 
                        attacker_vp, 
                        victim_weight
                    ).to(device)
            loss = loss1 + loss2
            loss.backward()

            losses1.append(loss1.item())
            losses2.append(loss2.item())

            # end of batch
            ###m1e4
        # adv_x = (x - alpha2*shaping*x.grad * lr/(batch_num1 + batch_num2)).detach_()
            adv_x = (x - alpha2*shaping*x.grad * lr/(batch_num1)).detach_()
            with torch.no_grad():
                    # print(x.grad[:, :, :3])

                    Ex = torch.sum(torch.pow(adv_x, 2))

                    if Ex>1/coeff**2:
                        adv_x = adv_x/(torch.sqrt(Ex)*coeff)
                    else:
                        adv_x = adv_x/1

                    adv_x = torch.round(adv_x*np.iinfo(np.short).max)\
                                    /np.iinfo(np.short).max
                    x = torch.clamp(orig_sd+adv_x, min=-1, max=1).detach_()\
                                - orig_sd

            # adv_x = (x - alpha2*shaping*x.grad.sign()/batch_num).detach_()
        print("batch: %d" % i, np.mean(losses1), np.mean(losses2))
        # print(loss1.item())
        losses.append((np.mean(losses1) + np.mean(losses2)) / 2) 
        # end of epoch
        with torch.no_grad():
            # print(x.grad[:, :, :3])

            Ex = torch.sum(torch.pow(adv_x, 2))

            if Ex>1/coeff**2:
                adv_x = adv_x/(torch.sqrt(Ex)*coeff)
            else:
                adv_x = adv_x/1

            adv_x = torch.round(adv_x*np.iinfo(np.short).max)\
                               /np.iinfo(np.short).max
            x = torch.clamp(orig_sd+adv_x, min=-1, max=1).detach_()\
                        - orig_sd
            early_stop.update(losses[-1], x)     

            if (losses[-1]) < loss_ub:
                print('epoch: {}, loss achieved.'.format(i))
                break
            elif early_stop.early_stop():
                print("Decreasing learning rate to: ", lr / 2)
                lr /= 2
                early_stop.reset_counter()
                early_stop.stop = False
                continue
                # print('epoch: {}, early stopped.'.format(i))
                # x = early_stop.get_best()
                # break
            elif early_stop.get_counter() > sub_patience\
                                    and SNR_dec < SNR_dec_max:
                SNR_lb -= 1
                SNR_dec += 1
                coeff = 10**(0.05*SNR_lb)/torch.sqrt(Ess)
                early_stop.reset_counter()
                print('SNR_lb decreased: {}'.format(SNR_lb))


    with torch.no_grad():
        x_end = (orig_sd + x)[:, :, 8000:-8000]
        x_end_batch = torch.reshape(
                    x_end,\
                    (x_end.size(0)*x_end.size(1), x_end.size(2))\
                    )
        vp = au2voiceprint(x_end_batch, sr, model, device)
        hy = \
            {'loss': loss.item(),\
             'SNR': 10*torch.log10(torch.sum(torch.pow(orig_sd, 2))\
                                    /torch.sum(torch.pow(x, 2))).item(),\
             's_v1': (1-cosine_similarity2(vp, victim1_vp, 1e-6)).item(),\
             's_v2': (1-cosine_similarity2(vp, victim2_vp, 1e-6)).item(),\
             's_a': (1-cosine_similarity2(vp, attacker_vp, 1e-6)).item()}
        print('loss: ', hy['loss'])
        print("SNR(dB): ", hy['SNR'])
        print("1-simi to victim1: ", hy['s_v1'])
        print("1-simi to victim2: ", hy['s_v2'])
        print("1-simi to attacker: ", hy['s_a'])

    if verbose:
        print("Updated X with {} elements, mean {:0.2f}, std {:0.2f}, min {:0.2f}, max {:0.2f}"
            .format(x.shape[0], x.mean().item(), x.std().item(), x.min().item(), x.max().item()))

        if plot_figure:
            import matplotlib.pyplot as plt
            # An original song data
            plt.subplot(2,2,1)
            osd = orig_sd.detach().cpu().numpy()
            plt.plot(osd[0, 0, :])
            print('orig_sd: ({}, {})'.format(osd.min(), osd.max()))

            # Plot x after gradient updates
            plt.subplot(2,2,2)
            plt.plot(x.detach().cpu().numpy()[0, 0, :])

            # Plot changes in x[0][0]
            plt.subplot(2,2,3)
            plt.plot((orig_sd + x).detach().cpu().numpy()[0, 0, :])

            # Plot Losses
            plt.subplot(2,2,4)
            plt.plot(losses)
            # print(losses)
            import time
            plt.savefig(history+'/Changes_{}.png'\
                            .format(int(time.time())))

    return x.detach().cpu().numpy(), hy

def poison_seg(
    song_seg, 
    victim_vp,
    attacker_vp,
    victim_corpus,
    model,
    device, 
    home_path,
    sr = 1.6e3, 
    history = './',
    epoch = 2000, 
    SNR_lb = 20.0,
    alpha = 64/(2**16-1), 
    loss_ub = 0.25,
    victim_weight = 1,
    patience =20,
    sub_patience = 5,
    SNR_dec_max = 5,
    plot_figure = True,
    report_frequency = 50,
    verbose = False,
    x0 = None,
    seg_winlen = 25840,
    stride = 3200,
    RIR = None,
    DA_mu = 0.4,
    DA_sigma = 0.1,
    batch_size = 500, 
    lr = 1e4
):

    """
    Generate perturbation of PoiSong for digital backdoor attack 
    on the enrollment phase.
    
    Aruguments
    ----------
    song_seg : ndarray [1, seg]
        The carrier of PoiSong.
    victim_vp : torch.Tensor [1, 512]
        The average voiceprint of the victim.
    attacker_vp : torch.Tensor [1, 512]
        The average voiceprint of the attacker.
    victim_corpus : dict
        A corpus of victim.
    model : torch.nn.Module/torch.nn.DataParallel
        The front-end model. The encoder maps an utterance 
        to an embedding.
    device : torch.device
        The device used.
    home_path : str
        The working directory.
    sr : float (default: 1.6e3)
        The sampling rate.
    history : str (default: './')
        The path to save the training histories and figures.
    epoch : int (default: 2000)
        The number of epoch of training.
    SNR_lb : float (default: 20.0)
        The lower bound of SNR between song_seg and x.
    alpha : float (default: 64/(2**16-1))
        The learning rate.
    loss_ub : float (default: 0.25)
        The upperbound of loss. Training stops when the loss
        is less than the upperbound.
    victim_weight : float (default: 1.0)
        The weight of the distance from victim's voiceprint 
        in the loss function versus the from attacker's.
    patience : int (default: 20)
        The patience of the EarlyStop method. the training 
        ends when the loss function does not decrease for so
        many consecutive epochs.
    sub_patience : int (default: 5)
        SNR_lb decreases when the loss function does not 
        decrease for so many consecutive epochs.
    SNR_dec_max : int (default: 5)
        The maximum times that SNR_lb can decreases.
    plot_figure : bool (default: True)
        Whether plot figures.
    report_frequency : int (default: 50)
        The frequency reporting the loss values.
    verbose : bool (default: False)
        Whether generate a process bar and print the information
        of the result at the end of the training process. If
        verbose=False, then figures are not plotted. Setting
        plot_figure=True will not work.
    x0 : ndarray [1, seg] (default: None)
        A initial value of x. If x0=None, x is assigned a random
        array.
    seg_winlen: int (default: 25840)
        The sliding window length for forwarding.
    stride: int (default: 3200)
        The stride of sliding window for forwarding.
    
    Returns
    ----------
    x : ndarray [1, seg]
        return x.detach().cpu().numpy().
    hy : dict   {'loss', 'SNR', 's_v', 's_a'}
        return the history of training, including loss, SNR, 
        the distance from victim, the distance from attacker.
    """
    model.eval()

    # original song data
    orig_sd = song_seg.unsqueeze(0).to(device)*0.5
    print('orig_sd: ({:.3f}, {:.3f})'.format(orig_sd.min().item(),\
            orig_sd.max().item()))

    Ess = torch.sum(torch.pow(orig_sd, 2))
    coeff = 10**(0.05*SNR_lb)/torch.sqrt(Ess)
    shaping = orig_sd.abs()/orig_sd.abs().max()
    alpha2 = alpha/shaping.mean()
    print('alpha2: {} = alpha * {}'.format(alpha2.item(), 1/shaping.mean().item()))

    # initialization of x
    if x0 is None:
        x = torch.randn(orig_sd.size()).to(device)
    else:
        x = torch.tensor(x0).unsqueeze(0).to(device)

    if victim_corpus.size(0) != 0:
        vu = victim_corpus
        Ess = torch.sum(torch.pow(orig_sd, 2),\
                            dim = 2, keepdim=True).cpu()
        Evu = torch.sum(torch.pow(vu, 2),\
                            dim = 2, keepdim=True)

        vu = vu/torch.sqrt(Evu)*torch.sqrt(Ess*4)\
                    *(DA_mu+DA_sigma*torch.randn([vu.size(0), 1, 1]))\
                    *0.5

        # print('Evu:', torch.sqrt(Evu))
        # print('Ess:', torch.sqrt(Ess))
        print('victim_utte({}): ({}, {})'.format(
                                                tuple(vu.size()),
                                                vu.min().item(),
                                                vu.max().item()
                                                ))
        vu = vu.to(device)
        
        E1 = torch.sum(torch.pow(vu, 2), dim=2)
        E2 = torch.sum(torch.pow(0.2*orig_sd, 2))
        E3 = torch.sum(torch.pow(2*orig_sd, 2))
        print('SNR(v, 0.2*s)', 10*torch.log10(E1/E2)[:5,:])
        print('SNR(v, 2*s)', 10*torch.log10(E1/E3)[:5,:])
    else:
        vu = 0

    losses = []     # recording losses
    mkdir(history)

    # Optimization Epochs
    if verbose:
        t_range = tqdm(range(epoch))    # Display a process bar
    else:
        t_range = range(epoch)

    early_stop = EarlyStopping(patience, verbose=True, \
            parameter_save_path = home_path+'saved_parameters/')
    SNR_dec = 0     # The times of decreasing the SNR_lb
    N_subsegs = (x.size(-1)-seg_winlen)//stride # 

    batch_size = batch_size if batch_size<vu.size(0) else vu.size(0)
    batch_num = vu.size(0)//batch_size
    
    grad = torch.zeros(size=x.shape, device=device)
    for i in t_range:
        x.requires_grad = True
        
        
        
        for batch_idx in range(batch_num):
            x.requires_grad_(True)
            x_1 = orig_sd + x
            vu_batch = vu[batch_idx*batch_size:(batch_idx+1)*batch_size, :]
            x_rir0 = torch.vstack([0.2*x_1+vu_batch, 2*x_1+vu_batch])
           
            if RIR.size(0) != 0:     # Physical   
                RIR = RIR.to(device)
                x_rir = F.conv1d(x_rir0, RIR, 
                            padding='same',
                            dilation=1)
            
            else:                   # Digital
                x_rir = x_rir0
            
            # x_rir = x_rir.clamp(min=-1, max=1)

            # print('x_1({})\tx_rir0({})\tx_rir({})'.\
            #         format(x_1.shape, x_rir0.shape, x_rir.shape))

            x_rir_batch = torch.reshape(
                        x_rir, 
                        (x_rir.size(0)*x_rir.size(1), x_rir.size(2))
                        ) 
            x_stack = torch.vstack(
                        [x_rir_batch[:, j*stride:j*stride+seg_winlen]\
                        for j in range(N_subsegs)]\
                        )
            if i == 0 and batch_idx == 0:
                print('x_stack({})'.format(x_stack.shape))

            vp = au2voiceprint(x_stack, sr, model, device)
            model.zero_grad()
            loss = loss_func(
                        vp, 
                        victim_vp, 
                        attacker_vp, 
                        victim_weight
                    ).to(device)

            loss.backward()
            losses.append(loss.item())

            #x.grad.data[:,:,:8000].zero_()
            #x.grad.data[:,:,-8000:].zero_()

            # end of batch
            ###m1e4
            adv_x = (x - alpha2*shaping*x.grad * lr/batch_num).detach_()
            with torch.no_grad():
                # print(x.grad[:, :, :3])

                Ex = torch.sum(torch.pow(adv_x, 2))

                if Ex>1/coeff**2:
                    adv_x = adv_x/(torch.sqrt(Ex)*coeff)
                else:
                    adv_x = adv_x/1

                adv_x = torch.round(adv_x*np.iinfo(np.short).max)\
                                /np.iinfo(np.short).max
                x = torch.clamp(orig_sd+adv_x, min=-1, max=1).detach_()\
                            - orig_sd

        print("batch: %d" % i, loss.item(), flush=True)
        # end of epoch
        with torch.no_grad():
            # print(x.grad[:, :, :3])

            Ex = torch.sum(torch.pow(adv_x, 2))

            if Ex>1/coeff**2:
                adv_x = adv_x/(torch.sqrt(Ex)*coeff)
            else:
                adv_x = adv_x/1

            adv_x = torch.round(adv_x*np.iinfo(np.short).max)\
                               /np.iinfo(np.short).max
            x = torch.clamp(orig_sd+adv_x, min=-1, max=1).detach_()\
                        - orig_sd
            early_stop.update(loss.item(), x)     

            if loss < loss_ub:
                print('epoch: {}, loss achieved.'.format(i))
                break
            elif early_stop.early_stop():
                print("Decreasing learning rate to: ", lr / 2)
                lr /= 2
                early_stop.reset_counter()
                early_stop.stop = False
                continue
                # print('epoch: {}, early stopped.'.format(i))
                # x = early_stop.get_best()
                # break
            elif early_stop.get_counter() > sub_patience\
                                    and SNR_dec < SNR_dec_max:
                SNR_lb -= 1
                SNR_dec += 1
                coeff = 10**(0.05*SNR_lb)/torch.sqrt(Ess)
                early_stop.reset_counter()
                print('SNR_lb decreased: {}'.format(SNR_lb))


    with torch.no_grad():
        x_end = (orig_sd + x)[:, :, 8000:-8000]
        x_end_batch = torch.reshape(
                    x_end,\
                    (x_end.size(0)*x_end.size(1), x_end.size(2))\
                    )
        vp = au2voiceprint(x_end_batch, sr, model, device)
        hy = \
            {'loss': loss.item(),\
             'SNR': 10*torch.log10(torch.sum(torch.pow(orig_sd, 2))\
                                    /torch.sum(torch.pow(x, 2))).item(),\
             's_v': (1-cosine_similarity2(vp, victim_vp, 1e-6)).item(),\
             's_a': (1-cosine_similarity2(vp, attacker_vp, 1e-6)).item()}
        print('loss: ', hy['loss'])
        print("SNR(dB): ", hy['SNR'])
        print("simi to victim: ", hy['s_v'])
        print("simi to attacker: ", hy['s_a'])

    if verbose:
        print("Updated X with {} elements, mean {:0.2f}, std {:0.2f}, min {:0.2f}, max {:0.2f}"
            .format(x.shape[0], x.mean().item(), x.std().item(), x.min().item(), x.max().item()))

        if plot_figure:
            import matplotlib.pyplot as plt
            # An original song data
            plt.subplot(2,2,1)
            osd = orig_sd.detach().cpu().numpy()
            plt.plot(osd[0, 0, :])
            print('orig_sd: ({}, {})'.format(osd.min(), osd.max()))

            # Plot x after gradient updates
            plt.subplot(2,2,2)
            plt.plot(x.detach().cpu().numpy()[0, 0, :])

            # Plot changes in x[0][0]
            plt.subplot(2,2,3)
            plt.plot((orig_sd + x).detach().cpu().numpy()[0, 0, :])

            # Plot Losses
            plt.subplot(2,2,4)
            plt.plot(losses)
            # print(losses)
            import time
            plt.savefig(history+'/Changes_{}.png'\
                            .format(int(time.time())))

    return x.detach().cpu().numpy(), hy


def generate(dataset, attacker_id, victim_id, sound_index=0, SNR_sx=10, nr_of_vu=50, segment=(40,41), victim_weight=0.5, shuffle=False,
            force_attacker=None, force_victim=None, RIR_num=0, batch_size = 500):
    DT = dataset
    if DT == 'LB':
        # people_list[0] is the fix role
        people_list = ['6930', '4077', '61', '260', '121', '1284', '2961']# 4077,61,260: M; 121,237,2961: F   # 6930, M
        attacker = people_list[attacker_id]
        victim1, victim2 = '4077', '2961'

        enrolled_speakers = ['237', '5105', '1580', '7176', '2300'] #'237'F, '5105'M, '1580'F, '7176'M, '2300'M
        dataset_path = '/home/jiangyi/Datasets/LibriSpeech/test-clean/'
        move_str = '/*/*.flac'
        mfb_ext = '.flac'
        utte_ext ='/*.flac'
        enroll_ext = move_str

    elif DT == 'VOX':
        #people_list = ['id10334', 'id10019', 'id10017', 'id10018', 'id10270', 'id10008', 'id10015']#16,17,18:M; 6,8,15:F
        people_list = ['id10334', 'id10019', 'id10017', 'id10018', 'id10469', 'id10470', 'id10473']#16,17,18:M; 6,8,15:F

        attacker = people_list[attacker_id]
        victim = people_list[victim_id]

        enrolled_speakers = ['id10025', 'id10022', 'id10023', 'id10013', 'id10014']# 1,2,3:M; 13,14:F
        dataset_path = '/home/jiangyi/Datasets/voxceleb_all/'
        move_str = '/*/*.wav'
        mfb_ext = '.wav'
        utte_ext = '/*.wav'
        enroll_ext = move_str

    elif DT == 'ST':
        people_list = ['P00016A', 'P00015A', 'P00017A', 'P00018A', 'P00015I', 'P00016I', 'P00017I']#16A, M
        attacker = people_list[attacker_id]
        victim = people_list[victim_id]

        enrolled_speakers = ['P00011A', 'P00011I', 'P00012A', 'P00012I', 'P00013A']# A:M; I:F
        dataset_path = '/home/jiangyi/Datasets/ST-CMDS-20170001_1-OS/'
        move_str = '20170001{}*.wav'
        mfb_ext = '.wav'
        utte_ext = '/*.wav'
        enroll_ext = '/*/*.wav'
    
    
    # attacker = force_attacker if force_attacker else attacker
    # victim = force_victim if force_victim else victim


    song_path = './carriers/'
    # song = [name for name in os.listdir(song_path) if name[-4:]=='flac'][sound_index]
    # for comparison
    song = 'Taylor Swift - Love Story.flac'
    # song = 'City_traffic.flac'
    assert os.path.exists(song_path + song)
    print(os.listdir(song_path))
    print('Song: ', song)

    from voxceleb_wav_reader import read_librispeech_structure
    from iden2 import DeepSpeakerIden
    from tqdm.notebook import tqdm
    import audio_processing as ap

    nrof_speakers = 5
    nrof_utte_each = 5

    resume = './checkpoint_37.pth'
    voiceprint_root=home_path+'lib_voiceprint/'

    # =================================================================================
    speakers = enrolled_speakers

    train_path = home_path + 'train/'
    train_path_v = home_path + 'train_v/'

    test_path = home_path + 'test/'
    victim_corpus = home_path + 'victim_corpus/'
    victim_enroll = home_path + 'victim_enroll/'
    attacker_path = home_path + 'attacker/'

    train_spks = read_librispeech_structure(train_path, True, '/*.npy')
    test_spks = read_librispeech_structure(test_path, True, '/*.npy')
    victim_spks = read_librispeech_structure(victim_corpus, True, '/*.npy')

    attacker_spks = read_librispeech_structure(attacker_path, True, '/*.npy')

    dsi = DeepSpeakerIden(device, nrof_utte_each, resume, 
                          voiceprint_root=voiceprint_root, 
                          enrolled_files = train_path, filelist = train_spks)
    dsi.build_voiceprint()

    song_data, sr = torchaudio.load(song_path+song)
    print("sample rate: ", sr)
    print("song size: ", song_data.size())
    print("duration: {}:{}".format(song_data.size(1)//sr//60, song_data.size(1)//sr%60))
    print("range: ({}, {})".format(song_data.min(), song_data.max()))
    # Bit depth: 16 bits

    import scipy.io as sio
    from torch.nn import functional as F
    import matplotlib.pyplot as plt

    all_RIR = sio.loadmat('./all_RIR_audio_1s.mat')['all_RIR']
    all_RIR_x = torch.FloatTensor(all_RIR[:, :8000-1]).unsqueeze(1)*20
    all_RIR_x = all_RIR_x[torch.randperm(all_RIR_x.size(0))]
    print('all_RIR.shape:', all_RIR_x.shape)

    from poisong import avg_voiceprint, cosine_similarity2
    import poisong
    import imp
    imp.reload(poisong)
    from poisong import avg_voiceprint, cosine_similarity2
    victim1_spks = [i for i in victim_spks if i['speaker_id'] == victim1]
    victim2_spks = [i for i in victim_spks if i['speaker_id'] == victim2]
    victim1_vp = avg_voiceprint(dsi, victim1_spks)
    victim2_vp = avg_voiceprint(dsi, victim2_spks)

    prematch1, prematch2 = dsi.match_voiceprint(victim1_vp, 5), dsi.match_voiceprint(victim2_vp, 5)
    print("victim 1 prematch: ", prematch1[0].tolist(), prematch1[1].tolist())
    print("victim 2 prematch: ", prematch2[0].tolist(), prematch2[1].tolist())

    attacker_vp = avg_voiceprint(dsi, attacker_spks)
    prematch = dsi.match_voiceprint(attacker_vp, 5)
    print("attacker prematch: ", prematch[0].tolist(), prematch[1].tolist())

    print('simi between victim1 and attacker:', cosine_similarity2(victim1_vp, attacker_vp, eps = 1.e-6).item())
    print('simi between victim2 and attacker:', cosine_similarity2(victim2_vp, attacker_vp, eps = 1.e-6).item())
    print('simi between victim1 and victim2 :', cosine_similarity2(victim1_vp, victim2_vp, eps = 1.e-6).item())

    from poisong_old import other_utte
    import poisong_old
    import imp
    imp.reload(poisong_old)
    from poisong_old import other_utte

    seg = 25840
    h_len = 8000
    # nr_of_vu = 50
    poisong_x = []
    histories = []

    for seg_no in tqdm(range(segment[0], segment[1])):
    # for seg_no in range(0):
        print("\nProcess: {}/{}".format(seg_no, song_data.size(1)//seg-1))

        song_seg = song_data[:, seg_no*seg-h_len:(seg_no+1)*seg+h_len]
        vu = other_utte([1, *song_seg.size()], victim_corpus, ext=utte_ext)# [0:nr_of_vu, :, :]
        vu1 = vu[victim1][0:nr_of_vu, :, :]
        vu2 = vu[victim2][0:nr_of_vu, :, :]

        ps_x, hy = poison_seg(
                        song_seg,
                        victim1_vp,
                        # victim2_vp,
                        attacker_vp,
                        vu1,
                        # vu2,
                        dsi.model,
                        dsi.device,
                        home_path = home_path,
                        sr = sr,
                        history = home_path+'/history/',
                        epoch = 500,
                        SNR_lb = SNR_sx,
                        alpha = 256/2**16, # 16
                        loss_ub = 0.25,
                        victim_weight = victim_weight,
                        patience = 20,
                        sub_patience = 5,
                        SNR_dec_max = 0,
                        plot_figure = True,
                        report_frequency = 100,
                        verbose = True,
                        x0 = None,
                        seg_winlen = 25840,
                        stride = 3200,
                        RIR = all_RIR_x[:RIR_num, :],
                        DA_mu=0.6,
                        DA_sigma=0.05,
                        batch_size = batch_size)

        poisong_x.append(song_seg[:, h_len:seg+h_len] + ps_x[0, :, h_len:seg+h_len])
        np.save(home_path+'history/'+'ps_x_seg_{}'.format(seg_no), ps_x)
        histories.append(hy)
        np.save(home_path+'history/'+'hy_seg_{}'.format(seg_no), hy)
        
    return locals()


# In[ ]:


def evaluate(package, N_p=5, SNR_vf=5, sound_index=0, segment=(40,41), victim_no=1):
    globals().update(package)
    print(dataset)
        
    from poisong import avg_voiceprint, cosine_similarity2, au2voiceprint
    import poisong
    import imp
    imp.reload(poisong)
    from poisong import avg_voiceprint, cosine_similarity2, au2voiceprint

    ################################
    if victim_no == 1:
        victim = victim1
        victim_vp = victim1_vp
        victim_spks = victim1_spks
    else:
        victim = victim2
        victim_vp = victim2_vp
        victim_spks = victim2_spks
    ################################
    
#     DT = dataset
#     if DT == 'LB':
#         dataset_path = '/home/usslab/Documents2/Jiangyi/LibriSpeech/test-clean/'
#         move_str = '/*/*.flac'
#         mfb_ext = '.flac'
#         utte_ext ='/*.flac'
#         enroll_ext = move_str

#     elif DT == 'VOX':
#         dataset_path = '/home/usslab/Documents2/xinfeng/dataset/voxceleb1/voxceleb_all/'
#         move_str = '/*/*.wav'
#         mfb_ext = '.wav'
#         utte_ext = '/*.wav'
#         enroll_ext = move_str

#     elif DT == 'ST':
#         dataset_path = '/home/usslab/Documents2/Jiangyi/ST-CMDS-20170001_1-OS/'
#         move_str = '20170001{}*.wav'
#         mfb_ext = '.wav'
#         utte_ext = '/*.wav'
#         enroll_ext = '/*/*.wav'
        
#     song_path = '/home/usslab/Documents2/Jiangyi/sound_effect_mono/'
#     song = [name for name in os.listdir(song_path) if name[-4:]=='flac'][sound_index]

#     print(os.listdir(song_path))
#     print('Song: ', song)
    
#     song_data, sr = torchaudio.load(song_path+song)
#     print("sample rate: ", sr)
#     print("song size: ", song_data.size())
#     print("duration: {}:{}".format(song_data.size(1)//sr//60, song_data.size(1)//sr%60))
#     print("range: ({}, {})".format(song_data.min(), song_data.max()))

    import wave
#     seg = 25840
#     h_len = 8000

    # other_dirs = ['PBIB_20211029_2/','PBIB_20211029_2_G3/','PBIB_20211029_2_G4/']
    # for od in other_dirs:
    #     ps_segs = glob.glob(home_path+'../'+od+'history/ps_seg_*.npy')
    #     for f in ps_segs:
    #         shutil.copy(f, home_path+'history/')

    # join all segs together
    poison_song = []
    for seg_no in range(segment[0], segment[1]):
        ps = np.load(home_path+'history/'+'ps_x_seg_{}.npy'.format(seg_no))
        poison_song.append(0.5*song_data[:, seg_no*seg:(seg_no+1)*seg] + ps[0, :, h_len:seg+h_len])
        print(ps.shape)

    poison_song_stacked = np.hstack([p[:,:] for p in poison_song])
    np.save(home_path+'poison_song.npy', poison_song_stacked)
    #np.save(home_path+'histories.npy', histories)

    poison_song_stacked = np.load(home_path+'poison_song.npy')
    print(poison_song_stacked.shape)
    
    global dsi
    vp = au2voiceprint(torch.tensor(poison_song_stacked), sr, dsi.model, device) # vp of song + poison noise only
    print('.npy FS simi to victim_vp:',(1-cosine_similarity2(vp, victim_vp, 1e-6)).item())
    print('.npy FS simi to attacker_vp:',(1-cosine_similarity2(vp, attacker_vp, 1e-6)).item())

    f = wave.open(home_path+'poison_song.flac', "wb")
    #set wav params
    f.setnchannels(1)
    f.setsampwidth(2)
    f.setframerate(c.SAMPLE_RATE)
    pss = (poison_song_stacked*np.iinfo(np.short).max).astype(np.short)
    f.writeframes(pss.tostring())
    f.close()

    test_song, _ = torchaudio.load(home_path+'poison_song.flac') # still vp of song + poison noise only, but audio
    vp = au2voiceprint(torch.tensor(test_song), sr, dsi.model, device)
    print('Audio FS simi to victim_vp:',(1-cosine_similarity2(vp, victim_vp, 1e-6)).item())
    print('Audio FS simi to attacker_vp:',(1-cosine_similarity2(vp, attacker_vp, 1e-6)).item())

    # ===============================================================================================
    import wave

    song_victim = 'song_victim_RIR({})/'.format(RIR_num)
    dirs = [home_path + song_victim, home_path + song_victim + victim]
    for d in dirs:    
        if not os.path.exists(d):
            os.mkdir(d)
        else:
            shutil.rmtree(d)
            os.mkdir(d)

    def SNR_nor(signal, noise, targetSNR=10):
        signal = torch.Tensor(signal)
        noise = torch.Tensor(noise)

        Evictim = torch.sum(torch.pow(signal, 2))
        coeff = 10**(0.05*targetSNR)/torch.sqrt(Evictim)
        Ex = torch.sum(torch.pow(noise, 2))
        noise2 = noise/(torch.sqrt(Ex)*coeff)

        print('coeff: ', 1/(torch.sqrt(Ex)*coeff))
        return noise2

    mSNR = SNR_vf
    victim_uttes = read_librispeech_structure(victim_enroll, True, utte_ext)
    victim_uttes = [i for i in victim_uttes if i['speaker_id'] == victim]
    for vu in victim_uttes:
        vu_data, vu_sr = torchaudio.load(vu['filename'])
        # vu_data /= vu_data.abs().max()
        shorter = vu_data.size(1) if vu_data.size(1) <= poison_song_stacked.shape[1] else poison_song_stacked.shape[1]

        vu_part = vu_data[:, :shorter]
        ps_part = poison_song_stacked[:, 0*seg:0*seg+shorter]

        ps_part2 = SNR_nor(vu_part, ps_part, mSNR)
        print('SNR:', 10*torch.log10(torch.sum(torch.pow(vu_part, 2))/torch.sum(torch.pow(torch.tensor(ps_part2), 2))).item())

        ps = (vu_part + ps_part2)
        ps = ps/ps.abs().max()

        print('vu size:', vu_data.size(), vu_data.max(), vu_data.min())
        print('ps size:', ps_part2.shape, ps_part2.max(), ps_part2.min())

        ps = ps.numpy()*np.iinfo(np.short).max

        vu_f = vu['filename'].split('/')
        vu_f[-3] = song_victim[:-1]
        print('/'.join(vu_f))

        f = wave.open('/'.join(vu_f), "wb")
        #set wav params
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(c.SAMPLE_RATE)
        ps = ps.astype(np.short)
        f.writeframes(ps.tostring())
        f.close()

        test_song, _ = torchaudio.load('/'.join(vu_f))
        vp = au2voiceprint(torch.tensor(test_song), sr, dsi.model, device)
        print((1-cosine_similarity2(vp, victim_vp, 1e-6)).item())
        print((1-cosine_similarity2(vp, attacker_vp, 1e-6)).item())

    def enroll(src, nrof_utte, dst, ext = '/*/*.flac'):
        files = glob.glob(src+ext)

        for idx, f in enumerate(files[nrof_utte[0]:nrof_utte[1]]):
            if not os.path.exists(dst):
                os.mkdir(dst)

            shutil.copy(f, dst)
            ap.torch_mk_MFB(dst+'/'+f.split('/')[-1], fbank_func = 0, trim = 0)

        print('Enroll: {}({})'.format(dst.split('/')[-1], len(files)))

    import time

    #song_victim = 'song_victim_20_cut/'
    enroll(home_path+'victim_enroll/', (0,5-N_p), train_path+victim, enroll_ext)
    enroll(home_path+song_victim, (5-N_p,5), train_path+victim, enroll_ext)

    dirs = [voiceprint_root]
    for d in dirs:    
        if not os.path.exists(d):
            os.mkdir(d)
        else:
            shutil.rmtree(d)
            os.mkdir(d)

    train_spks = read_librispeech_structure(train_path, True, '/*.npy')
    # print(train_spks)
    dsi = DeepSpeakerIden(device, nrof_utte_each, resume, 
                          voiceprint_root=voiceprint_root, 
                          enrolled_files = train_path, filelist = train_spks)
    dsi.build_voiceprint()

    print('victim: ',victim)
    print('attacker: ',attacker)

    v_ASR = [0,0,0] # OSI, CSI, SV
    a_ASR = [0,0,0]

    print(dsi.match_voiceprint(victim_vp, 5))
    print(dsi.match_voiceprint(attacker_vp, 5))

    thresh = 0.711
    for v_utte in victim_spks:
        for i in range(10):
            result = dsi.match_utte(v_utte, top_k = nrof_speakers+1)
            print(result[0].tolist(), result[1].tolist())
            if victim == result[0][0]:
                v_ASR[0] += 1

                if result[1][0]<thresh:
                    v_ASR[1] += 1

            if result[1][result[0].tolist().index(victim)] <thresh:
                v_ASR[2] += 1

    for a_utte in attacker_spks:
        for i in range(10):
            result = dsi.match_utte(a_utte, top_k = nrof_speakers+1)
            print(result[0].tolist(), result[1].tolist())
            if victim == result[0][0]:
                a_ASR[0] += 1

                if result[1][0]<thresh:
                    a_ASR[1] += 1

            if result[1][result[0].tolist().index(victim)] <thresh:
                a_ASR[2] += 1

    vn = len(victim_spks)
    an = len(attacker_spks)
    print(np.array(v_ASR).T/vn/10, v_ASR, vn*10)
    print(np.array(a_ASR).T/an/10, a_ASR, an*10)
    
    shutil.rmtree(train_path+victim)


package = generate(dataset='LB', attacker_id=0, victim_id=6, sound_index=0, SNR_sx=10, nr_of_vu=30, segment=(41,43), RIR_num=0, batch_size = 2, victim_weight=0.7)
evaluate(package=package, N_p=5, SNR_vf=5, sound_index=0, segment=(41,43))
