
import os
import numpy as np
import scipy.io as spio
import gzip
import pickle
from sklearn import cross_validation
from sklearn import preprocessing


_floatX = np.float32


def load_musk1_data(file_path, prepro_func=None):

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
        bag['inst_prob'] = np.zeros([n_instances, ])
        bag['starting_point'] = np.zeros([n_instances, ])
        bag_labels.append(bag['label'])

    if prepro_func is not None:
        bags = prepro_func(bags)

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
        feature.append(preprocess(segment))

    feature = np.asarray(feature).squeeze()

    pkl_filename = '%s_%s_feature.pkl.gz' % (target, data_type)
    with gzip.open(pkl_filename, 'wb') as f:
        pickle.dump(feature, f)

    if data_type == 'preictal':
        label = np.ones(shape=[feature.shape[0], feature.shape[1]], dtype=np.int8)
    elif data_type == 'interictal':
        label = np.zeros(shape=[feature.shape[0], feature.shape[1]], dtype=np.int8)
    else:
        label = None

    if label is not None:
        pkl_filename = '%s_%s_label.pkl.gz' % (target, data_type)
        with gzip.open(pkl_filename, 'wb') as f:
            pickle.dump(label, f)


def load_kaggle_data_into_instance(target, cv_ratio=None):
    data_type = 'preictal'
    filename = '%s_%s_feature.pkl.gz' % (target, data_type)
    with gzip.open(filename, 'r') as f:
        preictal_feature = pickle.load(f)

    filename = '%s_%s_label.pkl.gz' % (target, data_type)
    with gzip.open(filename, 'r') as f:
        preictal_label = pickle.load(f)

    d0, d1, d2 = preictal_feature.shape
    preictal_feature = preictal_feature.reshape(d0 * d1, d2)
    preictal_label = preictal_label.reshape(d0 * d1, 1)

    data_type = 'interictal'
    filename = '%s_%s_feature.pkl.gz' % (target, data_type)
    with gzip.open(filename, 'r') as f:
        interictal_feature = pickle.load(f)

    filename = '%s_%s_label.pkl.gz' % (target, data_type)
    with gzip.open(filename, 'r') as f:
        interictal_label = pickle.load(f)

    d0, d1, d2 = interictal_feature.shape
    interictal_feature = interictal_feature.reshape(d0 * d1, d2)
    interictal_label = interictal_label.reshape(d0 * d1, 1)

    feature = np.vstack((interictal_feature, preictal_feature))
    label = np.vstack((interictal_label, preictal_label)).squeeze()

    if cv_ratio is None:
        print('feature size is: ', feature.shape)
        print('label size is: ', label.shape)
        return feature, label
    else:
        x_train, x_test, y_train, y_test = cross_validation.train_test_split(feature, label,
                                                                             test_size=cv_ratio,
                                                                             random_state=0)
        print('training set size is: ', x_train.shape[0])
        print('testing set size is: ', x_test.shape[0])
        return x_train, y_train, x_test, y_test

