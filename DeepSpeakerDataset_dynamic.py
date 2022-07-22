# from __future__ import print_function
import numpy as np
import torch.utils.data as data


def find_classes(voxceleb):
    """
    find all classes/speakers, and give each speaker an simple index.
    args:      list: voxceleb = [dict: datum]
                        datum['speaker_id']
    return:     list: classes
                dict: class_to_idx """

    classes = list(set([datum['speaker_id'] for datum in voxceleb]))
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}

    return classes, class_to_idx

def create_indices(_features):
    """
        Create indices from _features to map a specific label to all of 
        its samples/files.

        args:       list: _features[(str: filepath, int: label)]

        returns:    dict: inds[int: label] = list[str: all_filepaths]
    """

    inds = dict()
    for idx, (feature_path, label) in enumerate(_features):
        if label not in inds:
            inds[label] = []
        inds[label].append(feature_path)
    return inds


def generate_triplets_call(indices, n_classes):
    """
        Generate triplets to train models.
        
        args:       dict: indices
                    - mapping labels to samples
        
                    int: n_classes
                    - the number of classes

        returns:    indices[c1][n1]
                        - anchor sample
                    indices[c1][n2]
                        - positive sample
                    indices[c2][n3]
                        - negative sample
                    c1  - positive label
                    c2  - negative label
    """

    # c1 is to be the anchor
    c1 = np.random.randint(0, n_classes)

    # c2 is to be the negative sample
    c2 = np.random.randint(0, n_classes)

    while len(indices[c1]) < 2:
        # if the class to be the anchor has less than 2 samples
        # then pick another class
        c1 = np.random.randint(0, n_classes)

    while c1 == c2:
        # if the negative class is the same as the anchor
        # then pick another class
        c2 = np.random.randint(0, n_classes)

    # pick 2 samples from the anchor class
    if len(indices[c1]) == 2:  # hack to speed up process
        n1, n2 = 0, 1
    else:
        # here may be wrong.
        # random.randint(low, high=None, size=None, dtype=int)
        # the high parameter will not be reach forever
        # n1 = np.random.randint(0, len(indices[c1]) - 1)
        # n2 = np.random.randint(0, len(indices[c1]) - 1)

        n1 = np.random.randint(0, len(indices[c1]))
        n2 = np.random.randint(0, len(indices[c1]))
        while n1 == n2:
            n2 = np.random.randint(0, len(indices[c1]))
    
    # if len(indices[c2]) ==1:
    #     n3 = 0
    # else:
    #     n3 = np.random.randint(0, len(indices[c2]) - 1)
    n3 = np.random.randint(0, len(indices[c2]))

    return [indices[c1][n1], indices[c1][n2], indices[c2][n3], c1, c2]


class DeepSpeakerDataset(data.Dataset):

    def __init__(self, voxceleb, vox_dir, n_triplets, loader, transform=None, *arg, **kw):

        print('Looking for audio [wav] files in {}.'.format(vox_dir))

        if len(voxceleb) == 0:
            raise(ValueError(('the length of voxceleb is zero')))

        # find_classes: e.g. [list: classes[str], dict[str: 'id10001']=0]
        classes, class_to_idx = find_classes(voxceleb)

        # features = [(the full path of a .wav file, idx of the speaker / label), ...]
        features = []
        for vox_item in voxceleb:
            item = (vox_dir + vox_item['filename'], class_to_idx[vox_item['speaker_id']])
            features.append(item)

        self.root = vox_dir
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.loader = loader

        # the number of Triplets generated
        self.n_triplets = n_triplets
        print('{} triplets to be generated.'.format(self.n_triplets))

        self.indices = create_indices(features)


    def __getitem__(self, index):
        '''
            Dynamically generating triplets. Saving memory.

            args:
                index: Index of the triplet or the matches - not of a single feature

            returns:

        '''
        def transform(feature_path):
            """Convert image into numpy array and apply transformation
               Doing this so that it is consistent with all other datasets
            """

            feature = self.loader(feature_path, '.'+feature_path.split('.')[-1])
            return self.transform(feature)

        # Get the index of each feature in the triplet
        a, p, n, c1, c2 = generate_triplets_call(self.indices, len(self.classes))
        # transform features if required
        feature_a, feature_p, feature_n = transform(a), transform(p), transform(n)
        return feature_a, feature_p, feature_n, c1, c2

    def __len__(self):
        return self.n_triplets

class DeepSpeakerTestset(data.Dataset):

    def __init__(self, voxceleb, vox_dir, loader, transform=None, *arg, **kw):

        print('Looking for audio [wav] files in {}.'.format(vox_dir))

        if len(voxceleb) == 0:
            raise(ValueError(('the length of voxceleb is zero')))

        # find_classes: e.g. [list: classes[str], dict[str: 'id10001']=0]
        classes, class_to_idx = find_classes(voxceleb)

        # features = [(the full path of a .wav file, idx of the speaker / label), ...]
        self.features = []
        for vox_item in voxceleb:
            item = (vox_dir + vox_item['filename'], vox_item['speaker_id'])
            self.features.append(item)

        self.root = vox_dir
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.loader = loader

        # the number of Triplets generated
        # self.n_triplets = n_triplets
        # print('{} triplets to be generated.'.format(self.n_triplets))

        # self.indices = create_indices(features)


    def __getitem__(self, index):
        '''
            Dynamically generating triplets. Saving memory.

            args:
                index: Index of the triplet or the matches - not of a single feature

            returns:

        '''
        def transform(feature_path):
            """Convert image into numpy array and apply transformation
               Doing this so that it is consistent with all other datasets
            """

            feature = self.loader(feature_path)
            return self.transform(feature)

        # Get the index of each feature in the triplet
        # a, p, n, c1, c2 = generate_triplets_call(self.indices, len(self.classes))
        # transform features if required
        a, class_a = self.features[index]
        feature_a = transform(a)
        return feature_a, class_a

    def __len__(self):
        return len(self.features)
        