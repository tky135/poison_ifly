import os, shutil
import time
import numpy as np 
import torch
import torchaudio
from model import cosine_similarity
from mkdir import mkdir
from tqdm.notebook import tqdm
from voxceleb_wav_reader import read_librispeech_structure
import constants as c

__version__='1.1'

def other_utte(x_size, speaker_path, ext, alpha = 1):
    ou = []

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
            su_data_pad[0, 0, :] = su_data[:, :x_size[-1]]

        ou.append(su_data_pad)

    return torch.cat(ou, 0)

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

def au2voiceprint(x, sr, model, device):
    '''
        x: [1, ?]
        0:25840-0:25999 produce 160 frames
    '''
    mfb_all = []
    for i in range(x.size(0)):
        mfb = torchaudio.compliance.kaldi.fbank(x[i:i+1, :], 
                            high_freq = sr/2,
                            low_freq = 0,
                            num_mel_bins = c.FILTER_BANK,
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
    loss = beta*(1-cosine_similarity2(vp, victim_vp, 1e-6)) \
        + (1-beta)*(1-cosine_similarity2(vp, attacker_vp, 1e-6))

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
    nr_of_vu = 50,
    seg_winlen = 25840,
    stride = 3200,
    RIR = None
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
    nr_of_vu : int (default: 1)
        The number of victim's utterances used for training.
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
    model.eval()    # evaluation mode

    # original song data
    orig_sd = song_seg.unsqueeze(0).to(device)*0.5
    print('orig_sd: ({:.3f}, {:.3f})'.format(orig_sd.min().item(),\
            orig_sd.max().item()))
    
    x_bound = orig_sd.abs()/10**(0.05*SNR_lb)
    x_bound = torch.floor(x_bound*np.iinfo(np.short).max)
    print('x_bound: ({:.3f}, {:.3f})/32767'.\
            format(x_bound.min().item(),\
                    x_bound.max().item()))
    x_bound /= np.iinfo(np.short).max

    # initialization of x
    if x0 is None:
        x = torch.randn(orig_sd.size()).to(device)
    else:
        x = torch.tensor(x0).unsqueeze(0).to(device)

    if nr_of_vu != 0:
        vu = victim_corpus
        Ess = torch.sum(torch.pow(orig_sd, 2),\
                            dim = 2, keepdim=True).cpu()
        Evu = torch.sum(torch.pow(vu, 2),\
                            dim = 2, keepdim=True)

        vu = vu/torch.sqrt(Evu)*torch.sqrt(Ess*4)\
                    *(0.4+0.1*torch.randn([vu.size(0), 1, 1]))\
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

    for i in t_range:
        x.requires_grad = True
        
        x_1 = orig_sd + x
        x_rir0 = torch.vstack([0.2*x_1+vu, 2*x_1+vu])
        
        if RIR is not None:     # Physical   
            x_rir = F.conv1d(x_rir0, RIR, 
                        padding='same',
                        dilation=1)
        
        else:                   # Digital
            x_rir = x_rir0
        
        x_rir = x_rir.clamp(min=-1, max=1)

        # print('x_1({})\tx_rir0({})\tx_rir({})'.\
        #         format(x_1.shape, x_rir0.shape, x_rir.shape))

        x_rir_batch = torch.reshape(
                    x_rir, 
                    (x_rir.size(0)*x_rir.size(1), x_rir.size(2))
                    ) 
        x_stack = torch.vstack(
                    [x_rir_batch[:, i*stride:i*stride+seg_winlen]\
                    for i in range(N_subsegs)]\
                    )
        if i == 0:
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

        adv_x = (x - alpha*x.grad.sign()).detach_()
        adv_x = torch.clamp(adv_x-x_bound, max=0) + x_bound
        adv_x = torch.clamp(adv_x+x_bound, min=0) - x_bound

        x = torch.clamp(orig_sd+adv_x, min=-1, max=1).detach_()\
                    - orig_sd
        early_stop.update(loss.item(), x)     

        if loss < loss_ub:
            print('epoch: {}, loss achieved.'.format(i))
            break
        elif early_stop.early_stop():
            print('epoch: {}, early stopped.'.format(i))
            x = early_stop.get_best()
            break
        elif early_stop.get_counter() > sub_patience\
                                and SNR_dec < SNR_dec_max:
            SNR_lb -= 1
            SNR_dec += 1

            # Adjust x_bound
            x_bound = orig_sd.abs()/10**(0.05*SNR_lb)
            x_bound = torch.floor(x_bound*np.iinfo(np.short).max)
            print('x_bound expanded: ({:.3f}, {:.3f})/32767'.\
                        format(x_bound.min().item(),\
                                x_bound.max().item()))
            x_bound /= np.iinfo(np.short).max

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