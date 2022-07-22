#!/usr/bin/env python
# coding: utf-8



# get_ipython().run_line_magic('pylab', 'inline')
import torch
from torchvision import transforms
import torchaudio

import os, glob, shutil
import numpy as np
from mkdir import mkdir

from model import cosine_similarity
from torch import nn, optim

import constants as c


# Set Devices
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="3"

# device = torch.device("cuda")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
print(device)
print(torch.cuda.device_count())

# home directory
home_path = "./NES/"
mkdir(home_path)




from poisong_old import EarlyStopping, au2voiceprint, loss_func, cosine_similarity2, loss_func_keep
from tqdm.notebook import tqdm
from ifly_ASV_2 import get_loss_fast
def poison_seg_NES(
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
    seg_no= -1
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
    

    # original song data (carrier)
    orig_sd = song_seg.unsqueeze(0).to(device)*0.5
    print('orig_sd: ({:.3f}, {:.3f})'.format(orig_sd.min().item(),            orig_sd.max().item()))
    print("orig_sd shape", orig_sd.shape)
    
    Ess = torch.sum(torch.pow(orig_sd, 2))
    coeff = 10**(0.05*SNR_lb)/torch.sqrt(Ess)
    shaping = orig_sd.abs()/orig_sd.abs().max()
    alpha2 = alpha/shaping.mean()
    print('alpha2: {} = alpha * {}'.format(alpha2.item(), 1/shaping.mean().item()))

    # initialization of x (attacker)
    if x0 is None:
        x = torch.randn(orig_sd.size()).to(device) 
    else:
        # raise Exception("break")
        x = torch.tensor(x0).to(device)
    # initialize to satisfy SNR_l
    Ex = torch.sum(torch.pow(x, 2))
    if Ex>1/coeff**2:
        x = x/(torch.sqrt(Ex)*coeff)

    if victim_corpus.size(0) != 0:
        vu = victim_corpus # [30, 1, D]
        Ess = torch.sum(torch.pow(orig_sd, 2), dim = 2, keepdim=True).cpu()
        Evu = torch.sum(torch.pow(vu, 2), dim = 2, keepdim=True)
        # print(torch.sqrt(Evu), torch.sqrt(Ess*4), DA_mu, 0.5)
        vu = vu/torch.sqrt(Evu)*torch.sqrt(Ess*4) * (DA_mu+DA_sigma*torch.randn([vu.size(0), 1, 1]))*0.5

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
    # print("vu shape", vu.shape)

    losses = []     # recording losses
    mkdir(history)
    # Optimization Epochs
    if verbose:
        t_range = range(epoch)   # Display a process bar
    else:
        t_range = range(epoch)

    early_stop = EarlyStopping(patience, verbose=True,             parameter_save_path = home_path+'saved_parameters/')
    SNR_dec = 0     # The times of decreasing the SNR_lb
    N_subsegs = (x.size(-1)-seg_winlen)//stride # 

    batch_size = batch_size if batch_size<vu.size(0) else vu.size(0)
    # print("batch_size: ", batch_size) # 2
    # number of batches per epoch, only update after each epoch
    batch_num = vu.size(0)//batch_size
    
    #NES
    # model.eval()    # evaluation mode

    #######################
    std = 0.002
    lr = 500 ## 5e2 ### 2e6
    samples_per_draw = 25

    #######################
    # std = 0.01
    grad = torch.zeros(size=x.shape, device=device)
    for b in t_range:
        # for each epoch
        ######################
        epoch_losses = []
        x.requires_grad = False
        # model.eval()
        
        ######################
        # x_1 = orig_sd + x
        for batch_idx in range(batch_num):
            if batch_idx == 7:
                continue
            # for each batch (for each vu sample)
            ####################################
            # get noises
            noise_pos = torch.normal(0, std, [samples_per_draw, *x.shape], device=device)
            noise = torch.vstack([torch.zeros([1, *x.shape], device=device), noise_pos, -noise_pos])


            noise_xs = noise + x + orig_sd # equivalent to x_1

            #####################################
            # victim sound for this batch
            # print("vu.shape", vu.shape) # [30, 1, 4]
            vu_batch = vu[batch_idx*batch_size:(batch_idx+1)*batch_size, :]     # [2, 1, D]
            # print('SNR_song_noise:', 10*torch.log10(torch.sum(torch.pow(0.2 * orig_sd, 2))/torch.sum(torch.pow(torch.tensor(0.2 * x), 2))).item())
            # print('SNR_victim_background: ', 10*torch.log10(torch.sum(torch.pow(vu_batch[1], 2))/torch.sum(torch.pow(0.2*noise_xs[0], 2))).item())
            # print("orig_sd: min %f, max %f, x: min %f, max %f, vu_batch: min %f, max %f" % (orig_sd.min(), orig_sd.max(), x.min(), x.max(), vu_batch.min(), vu_batch.max()))
            # iterate through 2 possible adjustments and different segments(2) in vu_batch

            adj_1 = [0.2*noise_xs+vu_batch[j] for j in range(vu_batch.shape[0])]
            # adj_1 = [vu_batch[j] for j in range(vu_batch.shape[0])]
            # adj_2 = [2*noise_xs+vu_batch[j] for j in range(vu_batch.shape[0])]
            # adj_1.extend(adj_2)     
            x_rir0 = torch.cat(adj_1, axis=1)  # [101, 4, 1, D]


            # if room impulse response provided, convolve
            if RIR.size(0) != 0:     # Physical   
                RIR = RIR.to(device)
                x_rir = F.conv1d(x_rir0, RIR,
                            padding='same',
                            dilation=1)
            
            else:                   # Digital
                x_rir = x_rir0

            # create subsegments, (j)
            x_stack = torch.cat(
                        [x_rir[:, :, :, j*stride:j*stride+seg_winlen]\
                        for j in range(N_subsegs)],\
                        axis=1
                        )  # [101, 20, 1, D']
            if b == 0 and batch_idx == 0:
                print('x_stack({})'.format(x_stack.shape))
            flat_x_stack = x_stack.view(-1, x_stack.shape[-1])

            ########################################################################
            ### GET LOSS FROM MODEL
            # with torch.no_grad():
            #     vp = au2voiceprint(flat_x_stack, sr, model, device) # [101x20, 512]
            # loss = loss_func_keep(
            #             vp, 
            #             victim_vp, 
            #             attacker_vp, 
            #             victim_weight
            #         ).to(device) # [101x20, 1]
            ########################################################################



            ### TODO make sure loss has the correct order
            ########################################################################
            ### GET LOSS FROM IFLY
            loss = get_loss_fast(flat_x_stack)
            # loss_alt = get_loss(flat_x_stack)
            # for i in range(loss.shape[0]):
            #     print(loss[i].item(), loss_alt[i].item())
            #     assert(loss[i].item() == loss_alt[i].item())
            ########################################################################



            loss = loss.reshape((x_stack.shape[0], x_stack.shape[1], x_stack.shape[2], 1)) # [101, 20, 1, 1]
            loss = torch.mean(loss, dim=1, keepdim=True).to(device) # [101, 1, 1, 1]
            # assuming the order is restored
            ########################################################################
            ### GET UPDATE_GRAD USING IFLY
            # print(loss.shape)
            update_grad = []
            if not torch.isnan(loss[0]):
                update_grad.append(loss[0].unsqueeze(0) * noise[0].unsqueeze(0))
            for i in range(1, samples_per_draw + 1):
                if torch.isnan(loss[i]) or torch.isnan(loss[i + samples_per_draw]):
                    continue
                else:
                    update_grad.append(loss[i].unsqueeze(0) * noise[i].unsqueeze(0))
                    update_grad.append(loss[i + samples_per_draw].unsqueeze(0) * noise[i + samples_per_draw].unsqueeze(0))
            # print(update_grad)
            if (len(update_grad) == 0):
                print("Empty list error for batch_idx = ", batch_idx)
                continue
            update_grad = torch.cat(update_grad, dim=0)
            ########################################################################



            ########################################################################
            ### GET UPDATE_GRAD USING MODEL
            # update_grad = loss * noise
            ########################################################################
            grad = 0.9 * grad + 0.1 * torch.mean(update_grad, axis=0) / std

            losses.append(loss[0].item())
            epoch_losses.append(loss[0].item())
            adv_x = (x - lr * alpha2*shaping*grad/batch_num).detach_()
            print("\t\t\t\t\t\tbatch_idx %d/%d" %(batch_idx, batch_num), "loss: ", loss[0].item(), flush=True)

            with torch.no_grad():
                Ex = torch.sum(torch.pow(adv_x, 2))

                if Ex>1/coeff**2:
                    adv_x = adv_x/(torch.sqrt(Ex)*coeff)
                else:
                    adv_x = adv_x/1

                adv_x = torch.round(adv_x*np.iinfo(np.short).max)                               /np.iinfo(np.short).max

                # update x
                x = torch.clamp(orig_sd+adv_x, min=-1, max=1).detach_()                        - orig_sd
                
                # # save x at the end of every epoch
                # np.save(home_path+'history/'+'ps_x_seg_ifly{}'.format(seg_no), x.cpu().numpy())
        early_stop.update(np.mean(epoch_losses), x)     

        if np.mean(epoch_losses) < loss_ub:
            print('epoch: {}, loss achieved.'.format(i))
            
            break
        elif early_stop.early_stop():
            if lr >= 50:
                print("Decreasing learning rate to: ", lr / 1.5)
                ## don't early stop, decrease lr instead
                lr /=2
            else:
                print("Not changing learning rate, lr = ", lr)
            # np.save(home_path+'history/'+'ps_x_seg_ifly{}'.format(seg_no), x.cpu().numpy())
            early_stop.reset_counter()
            early_stop.stop = False
            continue
            print('epoch: {}, early stopped.'.format(i))
            x = early_stop.get_best()
            break
        elif early_stop.get_counter() > sub_patience                                    and SNR_dec < SNR_dec_max:
            SNR_lb -= 1
            SNR_dec += 1
            coeff = 10**(0.05*SNR_lb)/torch.sqrt(Ess)
            early_stop.reset_counter()
            print('SNR_lb decreased: {}'.format(SNR_lb))
        print("Epoch: ", b, "NES: ", np.mean(epoch_losses), flush=True)
        # save x at the end of every epoch
        np.save(home_path+'history/'+'ps_x_seg_ifly{}'.format(seg_no), x.cpu().numpy())            
    hy =             {'loss': np.mean(epoch_losses),             'SNR': 10*torch.log10(torch.sum(torch.pow(orig_sd, 2))                                    /torch.sum(torch.pow(x, 2))).item()}
    return x.detach().cpu().numpy(), hy

    with torch.no_grad():
        x_end = (orig_sd + x)[:, :, 8000:-8000]
        x_end_batch = torch.reshape(
                    x_end,\
                    (x_end.size(0)*x_end.size(1), x_end.size(2))\
                    )
        vp = au2voiceprint(x_end_batch, sr, model, device)
        hy =             {'loss': loss[0].item(),             'SNR': 10*torch.log10(torch.sum(torch.pow(orig_sd, 2))                                    /torch.sum(torch.pow(x, 2))).item(),             's_v': (1-cosine_similarity2(vp, victim_vp, 1e-6)).item(),             's_a': (1-cosine_similarity2(vp, attacker_vp, 1e-6)).item()}
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
            plt.savefig(history+'/Changes_{}.png'                            .format(int(time.time())))

    return x.detach().cpu().numpy(), hy


# In[12]:


def generate(dataset, attacker_id, victim_id, sound_index=0, SNR_sx=10, nr_of_vu=50, segment=(40,41), victim_weight=0.5, shuffle=False,
            force_attacker=None, force_victim=None, RIR_num=0, batch_size = 500):
    DT = dataset
    if DT == 'LB':
        # people_list[0] is the fix role
        people_list = ['6930', '4077', '61', '260', '121', '1284', '2961']# 4077,61,260: M; 121,237,2961: F   # 6930, M
        attacker = people_list[attacker_id]
        victim = people_list[victim_id]

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
    
    
    attacker = force_attacker if force_attacker else attacker
    victim = force_victim if force_victim else victim

    
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

    # raise Exception("break")
    from poisong import avg_voiceprint, cosine_similarity2
    import poisong
    import imp
    imp.reload(poisong)
    from poisong import avg_voiceprint, cosine_similarity2

    # victim_vp = avg_voiceprint(dsi, victim_spks)
    # prematch = dsi.match_voiceprint(victim_vp, 5)
    # print(prematch[0].tolist(), prematch[1].tolist())

    # attacker_vp = avg_voicepkrint(dsi, attacker_spks)
    # prematch = dsi.match_voiceprint(attacker_vp, 5)
    # print(prematch[0].tolist(), prematch[1].tolist())

    # print('simi between victim and attacker:', cosine_similarity2(victim_vp, attacker_vp, eps = 1.e-6).item())


    import poisong_old
    import imp
    imp.reload(poisong_old)
    from poisong_old import other_utte

    seg = 25840
    # seg = 10000
    h_len = 8000
    # nr_of_vu = 50
    poisong_x = []
    histories = []

    # for seg_no in range(segment[0], segment[1]):
    for seg_no in range(2, 3):

        print("\nProcess: {}/{}".format(seg_no, song_data.size(1)//seg-1))

        song_seg = song_data[:, seg_no*seg-h_len:(seg_no+1)*seg+h_len]
        vu = other_utte([1, *song_seg.size()], victim_corpus, ext=utte_ext)[0:nr_of_vu, :, :]

        ps_x, hy = poison_seg_NES(
                        
                        song_seg,
                        None,
                        None,
                        vu,
                        None,
                        torch.device("cpu"),
                        seg_no = seg_no,
                        home_path = home_path,
                        sr = sr,
                        history = home_path+'/history/',
                        epoch = 1000,
                        SNR_lb = SNR_sx,
                        alpha = 256/2**16,  # 256/2**16
                        loss_ub = 0.25,
                        victim_weight = victim_weight,
                        patience = 10,   # 20, 5
                        sub_patience = 5,
                        SNR_dec_max = 0,
                        plot_figure = True,
                        report_frequency = 100,
                        verbose = True,
                        x0 = np.load(home_path+'history/'+'ps_x_seg_ifly{}.npy'.format(seg_no)),
                        seg_winlen = 25840,
                        stride = 3200,
                        RIR = all_RIR_x[:RIR_num, :],
                        DA_mu=0.6,
                        DA_sigma=0.05,
                        batch_size = batch_size)

        poisong_x.append(song_seg[:, h_len:seg+h_len] + ps_x[0, :, h_len:seg+h_len])
        # np.save(home_path+'history/'+'ps_x_seg_ifly{}'.format(seg_no), ps_x)
        histories.append(hy)
        np.save(home_path+'history/'+'hy_seg_ifly{}'.format(seg_no), hy)
        
    return locals()


# In[13]:

from ifly_ASV_2 import *
def evaluate(package, N_p=5, SNR_vf=5, sound_index=0, segment=(40,41)):
    globals().update(package)
    print(dataset)

    print('victim: ',victim)
    print('attacker: ',attacker)
    eval_ifly  = iFlytek_SV(enable_check=True, APIKey = "8ad9c4b60301c12b0d02b7c45f07df14", APPId = "b87c286e", APISecret = "M2FiYWFlYjY5NDAyNjk4OGI2ZDE2OWMz")
    eval_ifly.create_group(groupId="eval", groupName="eval", groupInfo="eval")
    # join all segs together
    poison_song = []
    # for seg_no in range(segment[0], segment[1]):
    song_part = None
    for seg_no in range(2,3):
        ps = np.load(home_path+'history/'+'ps_x_seg_ifly{}.npy'.format(seg_no))
        song_part = 0.5 * song_data[:, seg_no*seg:(seg_no+1)*seg]
        noise_part = 1 * ps[0, :, h_len:seg+h_len]
        # poison_song.apped(0.5 * song_data[:, seg_no * seg])
        # ps_part2 = SNR_nor(vu_part, ps_part, 10)
        poison_song.append(song_part + noise_part)
        print(song_part.shape, noise_part.shape)
        print('SNR_song_noise:', 10*torch.log10(torch.sum(torch.pow(song_part, 2))/torch.sum(torch.pow(torch.tensor(noise_part), 2))).item())

    poison_song_stacked = np.hstack([p[:,:] for p in poison_song])
    np.save(home_path+'poison_song.npy', poison_song_stacked)
    # np.load
    print(poison_song_stacked.shape)
    # raise Exception("break")
    torchaudio.save(filepath="poison_song.mp3", src=torch.tensor(poison_song_stacked), sample_rate=16000,  channels_first=True, compression=-4.5, format="mp3")

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
    # DA_mu=0.6
    # DA_sigma=0.05
    victim_uttes = read_librispeech_structure(victim_enroll, True, utte_ext)

    for vu in victim_uttes[0:]:
        vu_data, vu_sr = torchaudio.load(vu['filename'])
        # vu_data = vu_data/torch.sqrt(Evu)*torch.sqrt(Ess*4) * (DA_mu)*0.5
        # vu_data /= vu_data.abs().max()

        shorter = vu_data.size(1) if vu_data.size(1) <= poison_song_stacked.shape[1] else poison_song_stacked.shape[1]

        vu_part = vu_data[:, :shorter]
        ps_part = poison_song_stacked[:, 0*seg:0*seg+shorter]

        ps_part2 = SNR_nor(vu_part, ps_part, mSNR)
        # ps_part2 = ps_part
        ps = (vu_part + ps_part2)
        print(vu_part.shape, ps_part2.shape)
        print('SNR_victim_background:', 10*torch.log10(torch.sum(torch.pow(vu_part, 2))/torch.sum(torch.pow(torch.tensor(ps_part2), 2))).item())
        ps = vu_part
        ps = ps/ps.abs().max()

        torchaudio.save(filepath="poison_song_victim.mp3", src=ps, sample_rate=16000,  channels_first=True, compression=-4.5, format="mp3")
        break

    # enroll poison_song_victim
    result = eval_ifly.create_feature(file_path="poison_song_victim.mp3", featureId="eval", featureInfo="eval")
    # result = eval_ifly.create_feature(file_path="mp3s/attacker.mp3", featureId="victim", featureInfo="victim")
    # result = eval_ifly.search_score_feature(file_path="mp3s/540.mp3", dstFeatureId="victim")
    # print(result)
    # result = eval_ifly.search_score_feature(file_path="mp3s/attacker.mp3", dstFeatureId="eval")
    # print(result)
    # raise Exception("break")
    v_ASR = [0,0,0] # OSI, CSI, SV
    a_ASR = [0,0,0]
    thresh = 0.711
    vu_eval = other_utte([1, 1, 41840], victim_corpus, ext=utte_ext)[nr_of_vu:, :, :] # nr_of_vu
    att_eval = other_utte([1, 1, 41840], attacker_path, ext=utte_ext)[:, :, :]
    victim_score = []
    att_score = []
    for victim_spk in vu_eval:
        victim_spk = victim_spk / max(victim_spk.max(), -victim_spk.min()) * np.iinfo(np.int16).max / (-np.iinfo(np.int16).min)
        torchaudio.save(filepath="temp.mp3", src=torch.tensor(victim_spk), sample_rate=16000,  channels_first=True, compression=-4.5, format="mp3")
        result = eval_ifly.search_score_feature("temp.mp3", dstFeatureId="eval")
        score = json.loads(result['payload']['searchScoreFeaRes']['text'].decode("UTF-8"))["score"]
        victim_score.append(score)
    print("attacker: ")
    for att_spk in att_eval:
        att_spk = att_spk / max(att_spk.max(), -att_spk.min()) * np.iinfo(np.int16).max / (-np.iinfo(np.int16).min)
        torchaudio.save(filepath="temp.mp3", src=torch.tensor(att_spk), sample_rate=16000,  channels_first=True, compression=-4.5, format="mp3")
        result = eval_ifly.search_score_feature("temp.mp3", dstFeatureId="eval")
        score = json.loads(result['payload']['searchScoreFeaRes']['text'].decode("UTF-8"))["score"]
        att_score.append(score)

    # vn = len(victim_spks)
    # an = len(attacker_spks)
    # print(np.array(v_ASR).T/vn/10, v_ASR, vn*10)
    # print(np.array(a_ASR).T/an/10, a_ASR, an*10)
    threshold = 0.4
    print("victim score: mean: ", np.mean(victim_score), "success rate: ", np.mean(np.array(victim_score) > threshold))
    print("attacker score: mean: ", np.mean(att_score), "success rate: ", np.mean(np.array(att_score) > threshold))
    raise Exception("break")
    shutil.rmtree(train_path+victim)



package = generate(dataset='LB', attacker_id=0, victim_id=6, sound_index=0, SNR_sx=10, nr_of_vu=30, segment=(41,43), RIR_num=0, batch_size = 2, victim_weight=0.7)



evaluate(package=package, N_p=5, SNR_vf=10, sound_index=0, segment=(41,43))