import numpy as np

from sklearn import preprocessing

_floatX = np.float32
_intX = np.int8


def load_musk1_data(file_path):

    instances_info = list()

    instances_number = 0

    with open(file_path) as f:
        for line in f.readlines():
            bag = dict()
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
        # bag['instances'] = preprocessing.scale(bag['instances'], axis=1)

        bag['label'] = max(bag['label'])
        bag['prob'] = 0
        bag['inst_prob'] = np.zeros([n_instances, ])
        bag['starting_point'] = np.zeros([n_instances, ])
        bag_labels.append(bag['label'])

    return bags, bag_labels


def preprocess_musk1_data(bags):
    for bag in bags:
        bag['instances'] = preprocessing.scale(bag['instances'], axis=1)

    return bags
