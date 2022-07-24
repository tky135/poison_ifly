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
import random
import constants as c

from poisong_old import EarlyStopping, cosine_similarity2, au2voiceprint, loss_func
# Set Devices
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="0"
os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"
# device = torch.device("cuda")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
# print(device)
# print(torch.cuda.device_count())

# home directory
home_path = "./NES/"
mkdir(home_path)


def generate(dataset, attacker_id, victim_id, sound_index=0, SNR_sx=10, nr_of_vu=50, segment=(40,41), victim_weight=0.5, shuffle=False,
            force_attacker=None, force_victim=None, RIR_num=0, batch_size = 500):
    DT = dataset
    if DT == 'LB':
        # people_list[0] is the fix role
        people_list = ['6930', '4077', '61', '260', '121', '1284', '2961']# 4077,61,260: M; 121,237,2961: F   # 6930, M
        attacker = '6930'
        #################################################################
        ### change victim1
        victim1, victim2 = '2961', '2961'
        #################################################################

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
    # print(os.listdir(song_path))
    # print('Song: ', song)

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
    # print("sample rate: ", sr)
    # print("song size: ", song_data.size())
    # print("duration: {}:{}".format(song_data.size(1)//sr//60, song_data.size(1)//sr%60))
    # print("range: ({}, {})".format(song_data.min(), song_data.max()))
    # Bit depth: 16 bits

    import scipy.io as sio
    from torch.nn import functional as F
    import matplotlib.pyplot as plt

    all_RIR = sio.loadmat('./all_RIR_audio_1s.mat')['all_RIR']
    all_RIR_x = torch.FloatTensor(all_RIR[:, :8000-1]).unsqueeze(1)*20
    all_RIR_x = all_RIR_x[torch.randperm(all_RIR_x.size(0))]
    # # print('all_RIR.shape:', all_RIR_x.shape)

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
    # # print("victim 1 prematch: ", prematch1[0].tolist(), prematch1[1].tolist())
    # # print("victim 2 prematch: ", prematch2[0].tolist(), prematch2[1].tolist())

    attacker_vp = avg_voiceprint(dsi, attacker_spks)
    prematch = dsi.match_voiceprint(attacker_vp, 5)
    # # print("attacker prematch: ", prematch[0].tolist(), prematch[1].tolist())

    # # print('simi between victim1 and attacker:', cosine_similarity2(victim1_vp, attacker_vp, eps = 1.e-6).item())
    # # print('simi between victim2 and attacker:', cosine_similarity2(victim2_vp, attacker_vp, eps = 1.e-6).item())
    # # print('simi between victim1 and victim2 :', cosine_similarity2(victim1_vp, victim2_vp, eps = 1.e-6).item())

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

    # for seg_no in tqdm(range(segment[0], segment[1])):
    return locals()


# In[ ]:


def evaluate(package, N_p=5, SNR_vf=5, sound_index=0, segment=(40,41), victim_no=1):

    globals().update(package)
    # print(dataset)
        
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

#     # print(os.listdir(song_path))
#     # print('Song: ', song)
    
#     song_data, sr = torchaudio.load(song_path+song)
#     # print("sample rate: ", sr)
#     # print("song size: ", song_data.size())
#     # print("duration: {}:{}".format(song_data.size(1)//sr//60, song_data.size(1)//sr%60))
#     # print("range: ({}, {})".format(song_data.min(), song_data.max()))

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
        # print(ps.shape)

    poison_song_stacked = np.hstack([p[:,:] for p in poison_song])
    np.save(home_path+'poison_song.npy', poison_song_stacked)
    #np.save(home_path+'histories.npy', histories)

    poison_song_stacked = np.load(home_path+'poison_song.npy')
    # print(poison_song_stacked.shape)
    
    global dsi
    vp = au2voiceprint(torch.tensor(poison_song_stacked), sr, dsi.model, device) # vp of song + poison noise only
    # print('.npy FS simi to victim_vp:',(1-cosine_similarity2(vp, victim_vp, 1e-6)).item())
    # print('.npy FS simi to attacker_vp:',(1-cosine_similarity2(vp, attacker_vp, 1e-6)).item())

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
    # # print('Audio FS simi to victim_vp:',(1-cosine_similarity2(vp, victim_vp, 1e-6)).item())
    # # print('Audio FS simi to attacker_vp:',(1-cosine_similarity2(vp, attacker_vp, 1e-6)).item())

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

        # # print('coeff: ', 1/(torch.sqrt(Ex)*coeff))
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
        # # print('SNR:', 10*torch.log10(torch.sum(torch.pow(vu_part, 2))/torch.sum(torch.pow(torch.tensor(ps_part2), 2))).item())

        ps = (vu_part + ps_part2)
        ps = ps/ps.abs().max()

        # # print('vu size:', vu_data.size(), vu_data.max(), vu_data.min())
        # # print('ps size:', ps_part2.shape, ps_part2.max(), ps_part2.min())

        ps = ps.numpy()*np.iinfo(np.short).max

        vu_f = vu['filename'].split('/')
        vu_f[-3] = song_victim[:-1]
        # # print('/'.join(vu_f))

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
        # # print((1-cosine_similarity2(vp, victim_vp, 1e-6)).item())
        # # print((1-cosine_similarity2(vp, attacker_vp, 1e-6)).item())

    def enroll(src, nrof_utte, dst, ext = '/*/*.flac'):
        files = glob.glob(src+ext)

        for idx, f in enumerate(files[nrof_utte[0]:nrof_utte[1]]):
            if not os.path.exists(dst):
                os.mkdir(dst)

            shutil.copy(f, dst)
            ap.torch_mk_MFB(dst+'/'+f.split('/')[-1], fbank_func = 0, trim = 0)

        # # print('Enroll: {}({})'.format(dst.split('/')[-1], len(files)))

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
    # # print(train_spks)
    dsi = DeepSpeakerIden(device, nrof_utte_each, resume, 
                          voiceprint_root=voiceprint_root, 
                          enrolled_files = train_path, filelist = train_spks)
    dsi.build_voiceprint()

    # # print('victim: ',victim)
    # # print('attacker: ',attacker)

    v_ASR = [0,0,0] # OSI, CSI, SV
    a_ASR = [0,0,0]

    # # print(dsi.match_voiceprint(victim_vp, 5))
    # # print(dsi.match_voiceprint(attacker_vp, 5))

    thresh = 0.711
    for v_utte in victim_spks:
        for i in range(10):
            result = dsi.match_utte(v_utte, top_k = nrof_speakers+1)
            # # print(result[0].tolist(), result[1].tolist())
            if victim == result[0][0]:
                v_ASR[0] += 1

                if result[1][0]<thresh:
                    v_ASR[1] += 1

            if result[1][result[0].tolist().index(victim)] <thresh:
                v_ASR[2] += 1

    for a_utte in attacker_spks:
        for i in range(10):
            result = dsi.match_utte(a_utte, top_k = nrof_speakers+1)
            # # print(result[0].tolist(), result[1].tolist())
            if victim == result[0][0]:
                a_ASR[0] += 1

                if result[1][0]<thresh:
                    a_ASR[1] += 1

            if result[1][result[0].tolist().index(victim)] <thresh:
                a_ASR[2] += 1

    vn = len(victim_spks)
    an = len(attacker_spks)
    # # print(np.array(v_ASR).T/vn/10, v_ASR, vn*10)
    # # print(np.array(a_ASR).T/an/10, a_ASR, an*10)
    
    shutil.rmtree(train_path+victim)
    return np.array(v_ASR).T/vn/10, np.array(a_ASR).T/an/10

# reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
torch.use_deterministic_algorithms(True)

# generality
vv, aa = [], []
for i in range(10):
    package = generate(dataset='LB', attacker_id=0, victim_id=6, sound_index=0, SNR_sx=10, nr_of_vu=30, segment=(41,43), RIR_num=0, batch_size = 2, victim_weight=0.7)
    v_asr, a_asr = evaluate(package=package, N_p=5, SNR_vf=5, sound_index=0, segment=(41,43))
    vv.append(v_asr.reshape(1, 3))
    aa.append(a_asr.reshape(1, 3))
vv = np.concatenate(vv, axis=0)
aa = np.concatenate(aa, axis=0)
print(vv)
print(aa)
print(np.mean(vv, axis=0))
print(np.mean(aa, axis=0))
