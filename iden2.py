from voxceleb_wav_reader import read_voxceleb_set, voxceleb_dir
import audio_processing as ap

from DeepSpeakerDataset_dynamic import DeepSpeakerTestset
from model import DeepSpeakerModel, cosine_similarity

import torch
import torch.nn as nn
import torchvision.transforms as transforms

import os
from tqdm import tqdm
import numpy as np

# Hyperparameter Definitions
embedding_size = 512
# resume = './logs/dpsker_logs_(2021-07-15_21:40:08)/checkpoint_33.pth'
# resume = './logs/dpsker_logs_(2021-07-30_00:07:10)/checkpoint_10.pth'
# resume = './logs/dpsker_logs_(2021-08-02_00:29:46)/checkpoint_24.pth'
# resume = './logs/lib_dpsker_logs_(2021-08-09_17:04:14)/checkpoint_14.pth'
# resume = './logs/lib_dpsker_logs_(2021-08-12)/checkpoint_29.pth'

batch_size = 1
test_input_per_file = 10
# classes = 1251
classes = 2172
# _seed = 2
# np.random.seed(_seed)

# trainsform for testing
transform_T = transforms.Compose([
    ap.truncatedinputfromMFB(input_per_file=test_input_per_file),
    ap.totensor()
])

file_loader = ap.read_MFB

def remove_nan(out):
    judge = torch.isnan(out).int().sum(axis = -1)
    out = out[judge==0]
    return out


class DeepSpeakerIden():
    '''
        DeepSpeaker Model used for identification.
    '''
    def __init__(self, device, nrof_utte_each, resume, voiceprint_root='./voiceprint/',
                enrolled_files=None, filelist=None):
        # device
        self.device = device

        # load pretrain model
        m = DeepSpeakerModel(embedding_size=embedding_size,
                            num_classes=classes)
        self.model = torch.nn.DataParallel(m).to(self.device)
        self._load_pretrain_model(resume)

        self.root = voiceprint_root

        # if not enrolled_files:
        #     self.enrolled_files = np.array(os.listdir(self.root))
        # else:
        #     self.enrolled_files = np.array(enrolled_files)
        
        assert enrolled_files

        self.enrolled_files = enrolled_files
        self.spks = np.array(os.listdir(enrolled_files))
        self.spks.sort()

        self.nrof_utte_each = nrof_utte_each

        self.filelist = filelist

        self.model.eval()


    def _load_pretrain_model(self, resume):
        if resume:
            if os.path.isfile(resume):
                print('=> loading checkpoint {}'.format(resume))
                checkpoint = torch.load(resume)
                self.model.load_state_dict(checkpoint['state_dict'])
            else:
                print('=> no checkpoint found at {}'.format(resume))

    def match(self, out_1, out_2):
        '''
            args: x1, x2: [test_input_per_file, 512]
        '''
        
        out_1 = remove_nan(out_1)
        out_1_len = out_1.size(0)
        out_2 = remove_nan(out_2)
        out_2_len = out_2.size(0)
        
        shorter = out_1_len if out_1_len<=out_2_len else out_2_len
        
        out_1 = out_1[:shorter,:].reshape(shorter, 1, out_1.size(1))
        out_2 = out_2[:shorter,:].reshape(shorter, 1, out_2.size(1))
        
        dists = 1-cosine_similarity(out_1, out_2, eps=1.e-6) # torch.sqrt(torch.sum((out_a - out_p) ** 2, 1))  # euclidean distance
        dists = dists.data.cpu().numpy()
        # print('dist: ', dists)
        # dists = dists.mean()
    
        return dists

    def match_utte(self, utte, top_k = 1):
        x_out = self.voiceprint(utte, '')
        # print(x_out.size())
        # print(x_out[:5, :5])
        
        # print(x_out.size())
        # x_out = x_out.repeat(self.nrof_utte_each, 1)
        # # print(x_out.size())

        # x_out = remove_nan(x_out)
        # x_out_len = x_out.size(0)

        # matches = np.zeros(len(self.enrolled_files))

        # with torch.no_grad():
        #     for idx, f in (enumerate(self.enrolled_files)):
        #         embeddings_enrolled = torch.Tensor(np.load(self.root+f)).to(self.device)
        #         dist = self.match(x_out, embeddings_enrolled)
        #         matches[idx] = dist

        # return self.enrolled_files[np.argsort(matches)[0:top_k]], \
        #         matches[np.argsort(matches)[0:top_k]]
        return self.match_voiceprint(x_out, top_k)

    def match_voiceprint(self, x_out, top_k = 1):
        '''
            args:   x_out: (1, 512)
        '''
        vp1_repeat = x_out.repeat(100, 1)

        all_dists = []
        for fb in self.spks:
            vpn = torch.Tensor(self.read_voiceprint({'speaker_id': fb})).to(self.device)
            dists = self.match(vp1_repeat, vpn)
            # print('dists: ', dists.shape)
            all_dists.append(np.mean(dists))

        all_dists = np.array(all_dists)

        return self.spks[np.argsort(all_dists)][:top_k], np.sort(all_dists)[:top_k]


    def __match_voiceprint(self, x_out, top_k = 30):
        # x_out = self.voiceprint(utte)
        # print(x_out.size())
        # print(x_out[:5, :5])
        
        # print(x_out.size())
        x_out = x_out.repeat(self.nrof_utte_each, 1)
        # print(x_out.size())

        x_out = remove_nan(x_out)
        x_out_len = x_out.size(0)

        matches = np.zeros(len(self.spks))
        all_results = []

        with torch.no_grad():
            for idx, f in (enumerate(self.spks)):
                embeddings_enrolled = torch.Tensor(np.load(self.root+f)).to(self.device)
                dist = self.match(x_out, embeddings_enrolled)
                all_results.append(dist)
                matches[idx] = dist.mean()

        argsort = np.argsort(matches)[0:top_k]
        all_results = np.array(all_results)
        print('enrolled files: ', self.spks)
        return self.spks[argsort], \
                matches[argsort], all_results[argsort]

    def voiceprint(self, utte, audio_root = voxceleb_dir):
        features = self.voicefeature(utte, audio_root)
        out = self.model(features)

        return out

    def __voiceprint(self, utte, audio_root = voxceleb_dir):
        self.model.eval()
        mfb = ap.read_MFB_flac(audio_root+utte['filename'])
        features = transform_T(mfb)
        features = features.reshape(features.size(0), 1, features.size(1), features.size(2)).to(self.device)
        # print(features.size())
        out = self.model(features)

        return out

    def voicefeature(self, utte, audio_root = voxceleb_dir):
        self.model.eval()
        mfb = ap.read_MFB_flac(audio_root+utte['filename'])
        features = transform_T(mfb)
        features = features.reshape(features.size(0), 1, features.size(1), features.size(2)).to(self.device)

        return features

    def build_voiceprint(self):
        self.model.eval()
        # voxceleb_dev = read_voxceleb_set('dev')
        # voxceleb_dev = read_librispeech_set('test')

        for idx, utte in tqdm(enumerate(self.filelist)):
            vp = self.voiceprint(utte, '')
            vp = vp.detach().cpu().numpy()
            
            if os.path.exists(self.root + utte['speaker_id'] + '.npy'):
                all_out = np.load(self.root + utte['speaker_id'] + '.npy')
                all_out = np.vstack([all_out, vp])
            else:
                all_out = vp
            
            # print(self.root+class_a[0]+'.npy')
            np.save(self.root+utte['speaker_id']+'.npy', all_out)
            

    def read_voiceprint(self, utte):
        return np.load(self.root+utte['speaker_id']+'.npy')

    def __build_voiceprint(self):
        '''
            (ABANDONED)
            Build voiceprint database.
        '''
        # train loader
        voxceleb_dev = read_voxceleb_set('dev')
        train_set = DeepSpeakerTestset(voxceleb = voxceleb_dev, vox_dir=voxceleb_dir,
                                        loader = file_loader, transform=transform_T)
        del voxceleb_dev

        train_loader = torch.utils.data.DataLoader(train_set, batch_size=1, 
                                                shuffle=False, pin_memory=True)

        self.model.eval()
 
        for idx, (data_a, class_a) in tqdm(enumerate(train_loader)): 
            current_sample = data_a.size(0)
            data_a = data_a.reshape(test_input_per_file *current_sample, 1, data_a.size(2), data_a.size(3))
            data_a = data_a.to(self.device)
            out = np.array(self.model(data_a).detach().cpu().numpy())
            
            if os.path.exists(self.root + class_a[0]+'.npy'):
                all_out = np.load(self.root + class_a[0]+'.npy')
                all_out = np.vstack([all_out, out])
            else:
                all_out = out
            
            # print(self.root+class_a[0]+'.npy')
            np.save(self.root+class_a[0]+'.npy', all_out)


if __name__=="__main__":
    # devices
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"]="1"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
    print('Device: ', device, 'Count: ', torch.cuda.device_count())

    dsi = DeepSpeakerIden(device, 10)

    # Build the voiceprint database
    # dsi.build_voiceprint()
    voxceleb_test = read_voxceleb_set('test')
    correct_count = 0
    for i in tqdm(voxceleb_test):
        f, m = dsi.match_voiceprint(i, top_k=10)

        if i['speaker_id']+'.npy' in f:
            correct_count += 1

    print('\33[91mACC: {:.8f}\n\33[0m'.format(correct_count/len(voxceleb_test)))
