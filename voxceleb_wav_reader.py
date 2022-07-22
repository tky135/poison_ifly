import pickle

voxceleb_dir = 'voxceleb/'
struc_all = 'data/vox_all'
struc_dev = 'data/vox_dev'
struc_test = 'data/vox_test'

librispeech_train = '/home/usslab/Documents2/Jiangyi/LibriSpeech/train-clean-360/'
librispeech_test = '/home/usslab/Documents2/Jiangyi/LibriSpeech/test-clean/'
ls_struc_train = 'data/ls_train'
ls_struc_test = 'data/ls_test'

def read_voxceleb_set(subset='all'):
    '''
        Read the structure saved with save_voxceleb_structure().
        args:   str: subset
                - 'dev' or 'test'

        returns:    list[dict]: voxceleb
    '''
    r = []

    if subset == 'dev':
        f = open(struc_dev, 'rb')
    elif subset == 'test':
        f = open(struc_test, 'rb')
    elif subset == 'all':
        f = open(struc_all, 'rb')
    else:
        raise ValueError("subset is not 'all', 'dev' or 'test'")

    r = pickle.load(f)
    f.close()

    return r

def read_librispeech_set(subset='train'):
    '''
        Read the structure saved with save_voxceleb_structure().
        args:   str: subset
                - 'train' or 'test'

        returns:    list[dict]: librispeech
    '''
    r = []

    if subset == 'train':
        f = open(ls_struc_train, 'rb')
    elif subset == 'test':
        f = open(ls_struc_test, 'rb')
    else:
        raise ValueError("subset is not 'train' or 'test'")

    r = pickle.load(f)
    f.close()

    return r

def save_voxceleb_structure(split_file='./iden_split.txt'):
    '''
        Save the structure read from read_voxceleb_structure().
        global args:    struc_dev
                        struc_test
    '''
    voxceleb = read_voxceleb_structure(split_file)
    test = [t for t in voxceleb if t['subset']=='test']
    dev = [t for t in voxceleb if t['subset']=='dev']

    print("Dumping training set...", end="")
    f = open(struc_dev, 'wb')
    pickle.dump(dev, f)
    f.close()
    print("\rTraining set, dumped.   ", flush=True)

    print("Dumping test set...", end="")
    f = open(struc_test, 'wb')
    pickle.dump(test, f)
    f.close()
    print("\rTest set, dumped.   ", flush=True)

    print("Dumping the whole set...", end="")
    f = open(struc_all, 'wb')
    pickle.dump(voxceleb, f)
    f.close()
    print("\rThe whole set, dumped.   ", flush=True)


def read_voxceleb_structure(split_file='./iden_split.txt'):
    '''
        Read the structure of the voxceleb directory from a file 
        indicating train-test spliting. Every line in this file should
        be seperated with a space, not '\t' or ',' etc.
        
        The directory should have the following form:
            e.g. xxx/id11120/3Um2w4UZCBw/00001.wav
        
        The iden_split.txt file can be found on the voxceleb website:
        [Dataset split for identification]: 
        https://www.robots.ox.ac.uk/~vgg/data/voxceleb/meta/iden_split.txt

        args:       str: split_file
        returns:    list[dict]: voxceleb
                    - voxceleb['speaker_id']    the unique id of speakers
                    - voxceleb['filename']      the path to the target audio file
                    - voxceleb['subset']        whether this file is belong to 
                                                    'dev' set or 'test' set
    '''

    voxceleb = []
    subset = ['dev', 'dev', 'test']

    with open(split_file, 'r') as f:
        row = f.readline()[:-1]
        while row:
            row = row.split(' ')
            voxceleb.append({'speaker_id':row[1][:7],  'filename':row[1], 'subset':subset[int(row[0])-1]})
            row = f.readline()[:-1]
    
    return voxceleb


def read_librispeech_structure(root=librispeech_train, verbose = False, ext = '/*/*.flac'):
    import os
    import glob
    subset = 'train'
    librispeech = []
    ids = os.listdir(root)
    print('{} IDs are found.'.format(len(ids)))
    for idx, _id in enumerate(ids):
        dirs = os.listdir(root+_id)
        # print('{} directories are found.'.format(len(dirs)))
        files = glob.glob(root+_id+ext)
        librispeech.extend([{'speaker_id': _id, 'filename': fn, 'subset':subset} for fn in files])
        
        if verbose:
            print('ID: {}, in {} dirs, found: {} .flac files'.format(_id, len(dirs), len(files)))
        # print(len(librispeech))
    
    return librispeech

def save_librispeech_structure(verbose = False):
    
    ls_train = read_librispeech_structure(librispeech_train)
    ls_test = read_librispeech_structure(librispeech_test)

    print("Dumping training set...", end="")
    f = open(ls_struc_train, 'wb')
    pickle.dump(ls_train, f)
    f.close()
    print("\rTraining set, dumped.   ", flush=True)
    
    print("Dumping test set...", end="")
    f = open(ls_struc_test, 'wb')
    pickle.dump(ls_test, f)
    f.close()
    print("\rTest set, dumped.   ", flush=True)