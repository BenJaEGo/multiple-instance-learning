
import os
import numpy as np
import scipy.io as spio
import gzip
import pickle

_floatX = np.float32
_intX = np.int8


def load_feature_from_txt(file_path):

    feature = list()
    label = list()
    with open(file_path) as f:
        for line in f.readlines():
            line_split = line.split(',')
            instance_attribute = np.asarray(line_split[0:-1], dtype=_floatX)
            instance_label = line_split[-1]
            feature.append(instance_attribute)
            label.append(instance_label)
    feature = np.asarray(feature)
    label = np.asarray(label, dtype=_intX)

    return feature, label


def load_musk_data(file_path):

    instances_info = list()

    instances_number = 0

    with open(file_path) as f:
        for line in f.readlines():
            line_split = line.split(',')
            bag_name = line_split[0]
            instance_attribute = np.asarray(line_split[1:-1], dtype=_floatX)
            instance_label = int(line_split[-1])
            instances_info.append((bag_name, instance_attribute, instance_label))
            instances_number += 1

    print('instance number in musk1 data set is: %d' % instances_number)

    bags = list()
    bag_name_list = list()

    for instance in instances_info:
        bag_name, instance_attribute, instance_label = instance

        bag_count = bag_name_list.count(bag_name)
        if bag_count:
            bag_idx = bag_name_list.index(bag_name)
            bags[bag_idx]['instances'].append(instance_attribute)
            bags[bag_idx]['label'].append(instance_label)
        else:
            bag_name_list.append(bag_name)
            bag = dict()
            bag['name'] = bag_name
            bag['instances'] = list()
            bag['instances'].append(instance_attribute)
            bag['label'] = list()
            bag['label'].append(instance_label)
            bags.append(bag)

    print('bag number in musk1 data set is: %d' % len(bags))
    bag_labels = list()

    for bag in bags:
        n_instances = len(bag['instances'])
        bag['instances'] = np.asarray(bag['instances'], dtype=_floatX)
        bag['label'] = max(bag['label'])
        bag['prob'] = 0
        bag['selected'] = 0
        bag['inst_prob'] = np.zeros([n_instances, ])
        bag['starting_point'] = np.zeros([n_instances, ])
        bag_labels.append(bag['label'])

    return bags, bag_labels


# generator to iterate over competition mat data
# target is Dog_1, Dog_2, and so on.
# data_type is interictal, preictal and test
def load_mat_data(data_dir, target, data_type):
    # ...\data\Dog_1
    dir = os.path.join(data_dir, target)
    # print(dir)
    done = False
    i = 0
    while not done:
        i += 1
        filename = '%s\%s_%s_segment_%0.4d.mat' % (dir, target, data_type, i)
        str_name = '%s_segment_%d' % (data_type, i)
        if os.path.exists(filename):
            mat_data = spio.loadmat(filename, struct_as_record=False, squeeze_me=True)
            data = mat_data[str_name]
            yield (data)
        else:
            if 1 == i:
                raise Exception("file %s not found" % filename)
            done = True


# data_type is one of ('preictal', 'interictal', 'test')
def parse_mat_data(data_dir, target, data_type, preprocess=None):

    feature = list()
    mat_data = load_mat_data(data_dir, target, data_type)
    for segment in mat_data:
        if preprocess:
            feature.append(preprocess(segment))
        else:
            raise NotImplementedError('save raw data to pickle files is useless.')

    feature = np.asarray(feature).squeeze()
    print(feature.shape)

    [n_bag, _, _] = feature.shape

    slice_idx = 0

    feature_dir = 'fft_time_corr_freq_corr'
    for idx in range(0, n_bag):
        pkl_filename = '%s\%s_%s_feature_%d.pkl.gz' % (feature_dir, target, data_type, slice_idx)
        with gzip.open(pkl_filename, 'wb') as f:
            pickle.dump(feature[idx], f, protocol=4)

        if data_type == 'preictal':
            label = np.ones(shape=[feature.shape[0], feature.shape[1]], dtype=_intX)
        elif data_type == 'interictal':
            label = np.zeros(shape=[feature.shape[0], feature.shape[1]], dtype=_intX)
        else:
            label = None

        if label is not None:
            pkl_filename = '%s\%s_%s_label_%d.pkl.gz' % (feature_dir, target, data_type, slice_idx)
            with gzip.open(pkl_filename, 'wb') as f:
                pickle.dump(label[idx], f, protocol=4)

        slice_idx += 1


def load_kaggle_data_into_instance(target):

    data_type = 'preictal'

    preictal_feature = list()
    preictal_label = list()

    slice_idx = 0
    done = False
    while not done:
        filename = '%s_%s_feature_%d.pkl.gz' % (target, data_type, slice_idx)
        if os.path.exists(filename):
            with gzip.open(filename, 'r') as f:
                feature_slice = pickle.load(f)
                preictal_feature.append(feature_slice)
        else:
            if 0 == slice_idx:
                raise Exception("file %s not found" % filename)
            done = True
        slice_idx += 1

    slice_idx = 0
    done = False
    while not done:
        filename = '%s_%s_label_%d.pkl.gz' % (target, data_type, slice_idx)
        if os.path.exists(filename):
            with gzip.open(filename, 'r') as f:
                feature_slice = pickle.load(f)
                preictal_label.append(feature_slice)
        else:
            if 0 == slice_idx:
                raise Exception("file %s not found" % filename)
            done = True
        slice_idx += 1

    preictal_feature = np.asarray(preictal_feature)
    preictal_label = np.asarray(preictal_label)

    data_type = 'interictal'
    interictal_feature = list()
    interictal_label = list()

    done = False
    slice_idx = 0
    while not done:
        filename = '%s_%s_feature_%d.pkl.gz' % (target, data_type, slice_idx)
        if os.path.exists(filename):
            with gzip.open(filename, 'r') as f:
                feature_slice = pickle.load(f)
                interictal_feature.append(feature_slice)
        else:
            if 0 == slice_idx:
                raise Exception("file %s not found" % filename)
            done = True
        slice_idx += 1

    done = False
    slice_idx = 0
    while not done:
        filename = '%s_%s_label_%d.pkl.gz' % (target, data_type, slice_idx)
        if os.path.exists(filename):
            with gzip.open(filename, 'r') as f:
                feature_slice = pickle.load(f)
                interictal_label.append(feature_slice)
        else:
            if 0 == slice_idx:
                raise Exception("file %s not found" % filename)
            done = True
        slice_idx += 1

    preictal_feature = np.asarray(preictal_feature)
    preictal_label = np.asarray(preictal_label)
    interictal_feature = np.asarray(interictal_feature)
    interictal_label = np.asarray(interictal_label)

    print('preictal feature shape: ', preictal_feature.shape)
    print('preictal label shape: ', preictal_label.shape)
    print('interictal feature shape: ', interictal_feature.shape)
    print('interictal label shape: ', interictal_label.shape)

    feature = np.vstack((interictal_feature, preictal_feature)).squeeze()
    label = np.vstack((interictal_label, preictal_label)).squeeze()

    print('return feature shape', feature.shape)
    print('return label shape', label.shape)

    return feature, label


def load_kaggle_data_into_bag(target):

    data_dir = 'fft_time_corr_freq_corr'

    data_type = 'preictal'
    preictal_feature = list()
    preictal_label = list()

    done = False
    slice_idx = 0
    while not done:
        filename = '%s\%s_%s_feature_%d.pkl.gz' % (data_dir, target, data_type, slice_idx)
        if os.path.exists(filename):
            with gzip.open(filename, 'r') as f:
                feature_slice = pickle.load(f)
                preictal_feature.append(feature_slice)
        else:
            if 0 == slice_idx:
                raise Exception("file %s not found" % filename)
            done = True
        slice_idx += 1

    done = False
    slice_idx = 0
    while not done:
        filename = '%s\%s_%s_label_%d.pkl.gz' % (data_dir, target, data_type, slice_idx)
        if os.path.exists(filename):
            with gzip.open(filename, 'r') as f:
                feature_slice = pickle.load(f)
                preictal_label.append(feature_slice)
        else:
            if 0 == slice_idx:
                raise Exception("file %s not found" % filename)
            done = True
        slice_idx += 1

    preictal_feature = np.asarray(preictal_feature)
    preictal_label = np.asarray(preictal_label)

    data_type = 'interictal'
    interictal_feature = list()
    interictal_label = list()

    done = False
    slice_idx = 0
    while not done:
        filename = '%s\%s_%s_feature_%d.pkl.gz' % (data_dir, target, data_type, slice_idx)
        if os.path.exists(filename):
            with gzip.open(filename, 'r') as f:
                feature_slice = pickle.load(f)
                interictal_feature.append(feature_slice)
        else:
            if 0 == slice_idx:
                raise Exception("file %s not found" % filename)
            done = True
        slice_idx += 1

    done = False
    slice_idx = 0
    while not done:
        filename = '%s\%s_%s_label_%d.pkl.gz' % (data_dir, target, data_type, slice_idx)
        if os.path.exists(filename):
            with gzip.open(filename, 'r') as f:
                feature_slice = pickle.load(f)
                interictal_label.append(feature_slice)
        else:
            if 0 == slice_idx:
                raise Exception("file %s not found" % filename)
            done = True
        slice_idx += 1

    preictal_feature = np.asarray(preictal_feature)
    preictal_label = np.asarray(preictal_label)
    interictal_feature = np.asarray(interictal_feature)
    interictal_label = np.asarray(interictal_label)

    print('preictal feature shape: ', preictal_feature.shape)
    print('preictal label shape: ', preictal_label.shape)
    print('interictal feature shape: ', interictal_feature.shape)
    print('interictal label shape: ', interictal_label.shape)

    bags = list()
    bag_labels = list()

    n_bag, n_instance_each_bag, n_feature = preictal_feature.shape

    for bag_idx in range(n_bag):
        bag = dict()
        bag['instances'] = preictal_feature[bag_idx]
        bag['label'] = int(np.max(preictal_label[bag_idx]))
        bag['prob'] = 0
        bag['selected'] = 0
        bag['inst_prob'] = np.zeros([n_instance_each_bag, ])
        bag['starting_point'] = np.zeros([n_instance_each_bag, ])

        bags.append(bag)
        bag_labels.append(bag['label'])

    n_bag, n_instance_each_bag, n_feature = interictal_feature.shape

    for bag_idx in range(n_bag):
        bag = dict()
        bag['instances'] = interictal_feature[bag_idx]
        bag['label'] = int(np.max(interictal_label[bag_idx]))
        bag['prob'] = 0
        bag['selected'] = 0
        bag['inst_prob'] = np.zeros([n_instance_each_bag, ])
        bag['starting_point'] = np.zeros([n_instance_each_bag, ])

        bags.append(bag)
        bag_labels.append(bag['label'])

    print('bag number in data set is: %d' % len(bags))

    return bags, bag_labels
