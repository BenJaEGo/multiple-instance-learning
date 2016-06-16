import numpy as np
from sklearn import preprocessing


def data_preprocess_musk_dd(bags):
    for bag in bags:
        # preprocess for musk data on dd method according to paper
        bag['instances'] /= 100
    return bags


def data_preprocess_musk_svm(bags):
    bag_labels = list()
    instances = list()
    for bag in bags:
        [n_instances, _] = bag['instances'].shape
        if 0 == bag['label']:
            bag['label'] = 0
            bag['inst_labels'] = np.zeros([n_instances, ])
            bag_labels.append(0)
        else:
            bag['label'] = 1
            bag['inst_labels'] = np.ones([n_instances, ])
            bag_labels.append(1)
        instances.extend(bag['instances'])
        # bag['instances'] /= 100
        # bag['instances'] = preprocessing.minmax_scale(bag['instances'], axis=1, feature_range=(-1, 1))
        # bag['instances'] = preprocessing.normalize(bag['instances'], axis=1)
    instances = np.asarray(instances)
    instances = preprocessing.minmax_scale(instances, axis=0, feature_range=(-1, 1))
    # instances = preprocessing.minmax_scale(instances, axis=0, feature_range=(0, 1))
    # instances = preprocessing.normalize(instances, norm='l2', axis=0)
    inst_idx = 0
    for bag in bags:
        [n_instances, _] = bag['instances'].shape
        bag['instances'] = instances[inst_idx: inst_idx + n_instances, :]
        inst_idx += n_instances
    return bags, bag_labels


def normalized_bag(bags):
    bag_labels = list()
    for bag in bags:
        [n_instances, _] = bag['instances'].shape
        if 0 == bag['label']:
            bag['inst_labels'] = np.zeros([n_instances, ])
            bag_labels.append(0)
        else:
            bag['inst_labels'] = np.ones([n_instances, ])
            bag_labels.append(1)
        bag['instances'] = preprocessing.normalize(bag['instances'], axis=1)
    return bags, bag_labels


def scale_bag(bags):
    bag_labels = list()
    for bag in bags:
        [n_instances, _] = bag['instances'].shape
        if 0 == bag['label']:
            bag['inst_labels'] = np.zeros([n_instances, ])
            bag_labels.append(0)
        else:
            bag['inst_labels'] = np.ones([n_instances, ])
            bag_labels.append(1)
        bag['instances'] = preprocessing.minmax_scale(bag['instances'], axis=1, feature_range=(0, 1))
    return bags, bag_labels


def normalized_inst(bags):
    bag_labels = list()
    instances = list()
    for bag in bags:
        [n_instances, _] = bag['instances'].shape
        if 0 == bag['label']:
            bag['inst_labels'] = np.zeros([n_instances, ])
            bag_labels.append(0)
        else:
            bag['inst_labels'] = np.ones([n_instances, ])
            bag_labels.append(1)
        instances.extend(bag['instances'])
    instances = np.asarray(instances)
    instances = preprocessing.normalize(instances, norm='l2', axis=1)
    inst_idx = 0
    for bag in bags:
        [n_instances, _] = bag['instances'].shape
        bag['instances'] = instances[inst_idx: inst_idx + n_instances, :]
        inst_idx += n_instances
    return bags, bag_labels


def scale_inst(bags):
    bag_labels = list()
    instances = list()
    for bag in bags:
        [n_instances, _] = bag['instances'].shape
        if 0 == bag['label']:
            bag['inst_labels'] = np.zeros([n_instances, ])
            bag_labels.append(0)
        else:
            bag['inst_labels'] = np.ones([n_instances, ])
            bag_labels.append(1)
        instances.extend(bag['instances'])
    instances = np.asarray(instances)
    instances = preprocessing.minmax_scale(instances, axis=1, feature_range=(0, 1))
    inst_idx = 0
    for bag in bags:
        [n_instances, _] = bag['instances'].shape
        bag['instances'] = instances[inst_idx: inst_idx + n_instances, :]
        inst_idx += n_instances
    return bags, bag_labels
