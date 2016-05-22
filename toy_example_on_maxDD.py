
import numpy as np
import random

from maxDD_inst import MaxDiverseDensity


def toy_example():

    dd_classifier = MaxDiverseDensity()

    train_bags = list()
    train_labels = list()

    n_dim = random.randint(5, 10)
    n_train_bags = random.randint(20, 100)

    n_positive_bag = 0
    n_negative_bag = 0
    idx = 0

    while idx < n_train_bags:
        bag = dict()
        if random.random() < 0.5:
            mu = 10
            sig = 1.0
            bag['label'] = 1
            train_labels.append(1)
            bag['prob'] = 0
            n_instances = np.random.randint(1, 20)
            bag['inst_prob'] = np.random.random([n_instances, ])
            bag['starting_point'] = np.zeros([n_instances, ])
            bag['instances'] = np.random.normal(mu, sig, [n_instances, n_dim])
            train_bags.append(bag)
            idx += 1
            n_positive_bag += 1
        else:
            mu = 0.0
            sig = 1.0
            bag['label'] = 0
            train_labels.append(0)
            bag['prob'] = 0
            n_instances = np.random.randint(1, 20)
            bag['inst_prob'] = np.random.random([n_instances, ])
            bag['starting_point'] = np.zeros([n_instances, ])
            bag['instances'] = np.random.normal(mu, sig, [n_instances, n_dim])
            train_bags.append(bag)
            idx += 1
            n_negative_bag += 1

    print('positive bags number is:', n_positive_bag)
    print('negative bags number is: ', n_negative_bag)

    idx = 0
    n_test_bags = random.randint(10, 30)
    test_bags = list()
    test_labels = list()

    while idx < n_test_bags:
        bag = dict()
        if random.random() > 0.5:
            mu = 10
            sig = 1.0
            bag['label'] = 1
            test_labels.append(1)
            bag['prob'] = 0
            n_instances = np.random.randint(1, 20)
            bag['inst_prob'] = np.random.random([n_instances, ])
            bag['starting_point'] = np.zeros([n_instances, ])
            bag['instances'] = np.random.normal(mu, sig, [n_instances, n_dim])
            test_bags.append(bag)
            idx += 1
        else:
            mu = 0.0
            sig = 1.0
            bag['label'] = 0
            test_labels.append(0)
            bag['prob'] = 0
            n_instances = np.random.randint(1, 20)
            bag['inst_prob'] = np.random.random([n_instances, ])
            bag['starting_point'] = np.zeros([n_instances, ])
            bag['instances'] = np.random.normal(mu, sig, [n_instances, n_dim])
            test_bags.append(bag)
            idx += 1

    print('test bags number is:', n_test_bags)

    targets, scales, fvals = dd_classifier.train(train_bags, scale_indicator=1, epochs=1)

    aggregate = 'max'
    threshold = 0.5

    p_bags_label, p_bags_prob, p_instances_label, p_instances_prob = dd_classifier.predict(targets, scales,
                                                                                           fvals, test_bags,
                                                                                           aggregate, threshold)
    print('testing accuracy is: %f' % (sum(test_labels == p_bags_label)/n_test_bags))


if __name__ == '__main__':
    toy_example()
