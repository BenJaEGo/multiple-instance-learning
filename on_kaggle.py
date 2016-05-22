
import time
from sklearn import cross_validation
from sklearn import preprocessing
import copy
import os
import numpy as np


from data_utils import parse_mat_data
from feature_utils import extract_fft_frequency_feature
from data_utils import load_kaggle_data_into_bag
from vis_utils import plot_roc


def data_preprocess_kaggle_dd(bags):
    for bag in bags:
        # bag['instances'] = preprocessing.minmax_scale(bag['instances'], feature_range=(0, 1), axis=1)
        bag['instances'] = preprocessing.normalize(bag['instances'], axis=1, norm='l2')
    return bags


def data_preprocess_musk_svm(bags):
    bag_labels = list()
    instances = list()
    for bag in bags:
        [n_instances, _] = bag['instances'].shape
        if 0 == bag['label']:
            bag['label'] = -1
            bag['inst_labels'] = np.ones([n_instances, ]) * -1
            bag_labels.append(-1)
        else:
            bag['label'] = 1
            bag['inst_labels'] = np.ones([n_instances, ])
            bag_labels.append(1)
        instances.extend(bag['instances'])
        # bag['instances'] = preprocessing.minmax_scale(bag['instances'], axis=1, feature_range=(-1, 1))
        # bag['instances'] = preprocessing.normalize(bag['instances'], axis=1)
    instances = np.asarray(instances)
    # print(instances.shape)
    # instances = preprocessing.minmax_scale(instances, axis=1, feature_range=(-1, 1))
    instances = preprocessing.minmax_scale(instances, axis=1, feature_range=(0, 1))
    # instances = preprocessing.normalize(instances, norm='l2', axis=1)
    inst_idx = 0
    for bag in bags:
        [n_instances, _] = bag['instances'].shape
        bag['instances'] = instances[inst_idx: inst_idx + n_instances, :]
        inst_idx += n_instances
    return bags, bag_labels


def extract_feature_on_kaggle_data():
    data_dir = r"E:\ZJU\kaggle\data"
    target = 'Dog_1'
    data_type = 'preictal'
    parse_mat_data(data_dir, target, data_type, extract_fft_frequency_feature)

    data_type = 'interictal'
    parse_mat_data(data_dir, target, data_type, extract_fft_frequency_feature)


def EMDD_inst_on_kaggle(split_ratio=None, cv_fold=None,
                       aggregate='min', threshold=0.5, scale_indicator=1, epochs=10):
    from EMDD_inst import EMDiverseDensity
    start_time = time.clock()
    dd_classifier = EMDiverseDensity()
    # target = 'Dog_1'
    target = 'Patient_1'
    bags, bag_labels = load_kaggle_data_into_bag(target)
    bags = data_preprocess_kaggle_dd(bags)

    if split_ratio is None and cv_fold is None:
        print('parameters setting: split_ratio = None, cv_fold = None, aggregate = %s, threshold = %f, '
              'scale_indicator = %d, epochs = %d' % (aggregate, threshold, scale_indicator, epochs))
        targets, scales, func_values = dd_classifier.train(bags, scale_indicator, epochs)
        p_bags_label, p_bags_prob, p_instances_label, p_instances_prob = dd_classifier.predict(targets, scales,
                                                                                               func_values, bags,
                                                                                               aggregate, threshold)
        accuracy = sum(bag_labels == p_bags_label) / len(bags)
        print('training accuracy is: %f' % accuracy)

        train_result = (targets, scales, func_values)
        predict_result = (p_bags_label, p_bags_prob, p_instances_label, p_instances_prob)
        data = (bags, bag_labels)
        end_time = time.clock()
        print(('The code for file ' + os.path.split(__file__)[1] + ' ran for %.1fs' % (end_time - start_time)))
        return data, train_result, predict_result

    elif split_ratio:
        print('parameters setting: split ratio = %f, cv_fold = None, aggregate = %s, '
              'threshold = %f, scale_indicator = %d, epochs = %d' %
              (split_ratio, aggregate, threshold, scale_indicator, epochs))
        train_bag, test_bag, train_label, test_label = cross_validation.train_test_split(bags,
                                                                                         bag_labels,
                                                                                         test_size=split_ratio,
                                                                                         random_state=10)

        targets, scales, func_values = dd_classifier.train(train_bag, scale_indicator, epochs)
        p_bags_label, p_bags_prob, p_instances_label, p_instances_prob = dd_classifier.predict(targets, scales,
                                                                                               func_values, test_bag,
                                                                                               aggregate, threshold)
        accuracy = sum(test_label == p_bags_label) / len(test_bag)
        print('split ratio is %f, testing accuracy is %f' % (split_ratio, accuracy))

        train_result = (targets, scales, func_values)
        predict_result = (p_bags_label, p_bags_prob, p_instances_label, p_instances_prob)
        data = (train_bag, train_label, test_bag, test_label)
        end_time = time.clock()
        print(('The code for file ' + os.path.split(__file__)[1] + ' ran for %.1fs' % (end_time - start_time)))
        return data, train_result, predict_result
    elif cv_fold:
        print('parameters setting: split_ratio = None, cv_fold = %d, aggregate = %s, '
              'threshold = %f, scale_indicator = %d, epochs = %d'
              % (cv_fold, aggregate, threshold, scale_indicator, epochs))
        accuracy_list = list()
        n_bags = len(bags)
        kf = cross_validation.KFold(n_bags, cv_fold, shuffle=True, random_state=0)
        cf = 1
        for train_idx, test_idx in kf:
            train_bag = list()
            train_label = list()
            for idx in train_idx:
                train_bag.append(copy.deepcopy(bags[idx]))
                train_label.append(bag_labels[idx])
            test_bag = list()
            test_label = list()
            for idx in test_idx:
                test_bag.append(copy.deepcopy(bags[idx]))
                test_label.append(bag_labels[idx])

            targets, scales, func_values = dd_classifier.train(train_bag, scale_indicator, epochs)
            p_bags_label, p_bags_prob, p_instances_label, p_instances_prob = dd_classifier.predict(targets, scales,
                                                                                                   func_values,
                                                                                                   test_bag,
                                                                                                   aggregate,
                                                                                                   threshold)
            accuracy = sum(test_label == p_bags_label) / len(test_bag)
            accuracy_list.append(accuracy)
            print('completed fold %d, accuracy is %f' % (cf, accuracy))
            cf += 1

        mean_accuracy = float(np.mean(np.asarray(accuracy_list)))
        print('mean accuracy with %d-fold cross validation is %f' % (cv_fold, mean_accuracy))
        end_time = time.clock()
        print(('The code for file ' + os.path.split(__file__)[1] + ' ran for %.1fs' % (end_time - start_time)))

        return accuracy_list
    else:
        pass


def EMDD_bag_on_kaggle(split_ratio=None, cv_fold=None, threshold=0.5, scale_indicator=1, epochs=10):
    from EMDD_bag import EMDiverseDensity
    start_time = time.clock()
    dd_classifier = EMDiverseDensity()
    target = 'Patient_1'
    bags, bag_labels = load_kaggle_data_into_bag(target)
    bags = data_preprocess_kaggle_dd(bags)
    if split_ratio is None and cv_fold is None:
        print('parameters setting: split_ratio = None, cv_fold = None, threshold = %f, '
              'scale_indicator = %d, epochs = %d' % (threshold, scale_indicator, epochs))
        targets, scales = dd_classifier.train(bags, scale_indicator, epochs, threshold)
        p_bags_label, p_bags_prob, p_instances_label, p_instances_prob = dd_classifier.predict(targets, scales,
                                                                                               bags,
                                                                                               threshold)
        accuracy = sum(bag_labels == p_bags_label) / len(bags)
        print('training accuracy is: %f' % accuracy)

        train_result = (targets, scales)
        predict_result = (p_bags_label, p_bags_prob, p_instances_label, p_instances_prob)
        data = (bags, bag_labels)
        end_time = time.clock()
        print(('The code for file ' + os.path.split(__file__)[1] + ' ran for %.1fs' % (end_time - start_time)))
        return data, train_result, predict_result

    elif split_ratio:
        print('parameters setting: split ratio = %f, cv_fold = None, '
              'threshold = %f, scale_indicator = %d, epochs = %d' %
              (split_ratio, threshold, scale_indicator, epochs))
        train_bag, test_bag, train_label, test_label = cross_validation.train_test_split(bags,
                                                                                         bag_labels,
                                                                                         test_size=split_ratio,
                                                                                         random_state=0)

        targets, scales = dd_classifier.train(train_bag, scale_indicator, epochs, threshold)
        p_bags_label, p_bags_prob, p_instances_label, p_instances_prob = dd_classifier.predict(targets, scales,
                                                                                               test_bag,
                                                                                               threshold)
        accuracy = sum(test_label == p_bags_label) / len(test_bag)
        print('split ratio is %f, testing accuracy is %f' % (split_ratio, accuracy))

        train_result = (targets, scales)
        predict_result = (p_bags_label, p_bags_prob, p_instances_label, p_instances_prob)
        data = (bags, bag_labels)
        end_time = time.clock()
        print(('The code for file ' + os.path.split(__file__)[1] + ' ran for %.1fs' % (end_time - start_time)))
        return data, train_result, predict_result
    elif cv_fold:
        print('parameters setting: split_ratio = None, cv_fold = %d, '
              'threshold = %f, scale_indicator = %d, epochs = %d'
              % (cv_fold, threshold, scale_indicator, epochs))
        accuracy_list = list()
        n_bags = len(bags)
        kf = cross_validation.KFold(n_bags, cv_fold, shuffle=True, random_state=0)
        cf = 1
        for train_idx, test_idx in kf:
            train_bag = list()
            train_label = list()
            for idx in train_idx:
                train_bag.append(copy.deepcopy(bags[idx]))
                train_label.append(bag_labels[idx])
            test_bag = list()
            test_label = list()
            for idx in test_idx:
                test_bag.append(copy.deepcopy(bags[idx]))
                test_label.append(bag_labels[idx])

            targets, scales = dd_classifier.train(train_bag, scale_indicator, epochs, threshold)
            p_bags_label, p_bags_prob, p_instances_label, p_instances_prob = dd_classifier.predict(targets, scales,
                                                                                                   test_bag,
                                                                                                   threshold)
            accuracy = sum(test_label == p_bags_label) / len(test_bag)
            accuracy_list.append(accuracy)
            print('completed fold %d, accuracy is %f' % (cf, accuracy))
            cf += 1

        mean_accuracy = float(np.mean(np.asarray(accuracy_list)))
        print('mean accuracy with %d-fold cross validation is %f' % (cv_fold, mean_accuracy))
        end_time = time.clock()
        print(('The code for file ' + os.path.split(__file__)[1] + ' ran for %.1fs' % (end_time - start_time)))

        return accuracy_list
    else:
        pass


def maxDD_on_kaggle(split_ratio=None, cv_fold=None, aggregate='min', threshold=0.5, scale_indicator=1, epochs=10):
    from maxDD_inst import MaxDiverseDensity

    start_time = time.clock()
    dd_classifier = MaxDiverseDensity()
    target = 'Patient_1'
    bags, bag_labels = load_kaggle_data_into_bag(target)
    bags = data_preprocess_kaggle_dd(bags)
    if split_ratio is None and cv_fold is None:
        print('parameters setting: split_ratio = None, cv_fold = None, aggregate = %s, threshold = %f, '
              'scale_indicator = %d, epochs = %d' % (aggregate, threshold, scale_indicator, epochs))
        targets, scales, func_values = dd_classifier.train(bags, scale_indicator, epochs)
        p_bags_label, p_bags_prob, p_instances_label, p_instances_prob = dd_classifier.predict(targets, scales,
                                                                                               func_values, bags,
                                                                                               aggregate, threshold)
        accuracy = sum(bag_labels == p_bags_label) / len(bags)
        print('training accuracy is: %f' % accuracy)

        train_result = (targets, scales, func_values)
        predict_result = (p_bags_label, p_bags_prob, p_instances_label, p_instances_prob)
        data = (bags, bag_labels)
        end_time = time.clock()
        print(('The code for file ' + os.path.split(__file__)[1] + ' ran for %.1fs' % (end_time - start_time)))
        return data, train_result, predict_result

    elif split_ratio:
        print('parameters setting: split ratio = %f, cv_fold = None, aggregate = %s, '
              'threshold = %f, scale_indicator = %d, epochs = %d' %
              (split_ratio, aggregate, threshold, scale_indicator, epochs))
        train_bag, test_bag, train_label, test_label = cross_validation.train_test_split(bags,
                                                                                         bag_labels,
                                                                                         test_size=split_ratio,
                                                                                         random_state=0)

        targets, scales, func_values = dd_classifier.train(train_bag, scale_indicator, epochs)
        p_bags_label, p_bags_prob, p_instances_label, p_instances_prob = dd_classifier.predict(targets, scales,
                                                                                               func_values, test_bag,
                                                                                               aggregate, threshold)
        accuracy = sum(test_label == p_bags_label) / len(test_bag)
        print('split ratio is %f, testing accuracy is %f' % (split_ratio, accuracy))

        train_result = (targets, scales, func_values)
        predict_result = (p_bags_label, p_bags_prob, p_instances_label, p_instances_prob)
        data = (bags, bag_labels)
        end_time = time.clock()
        print(('The code for file ' + os.path.split(__file__)[1] + ' ran for %.1fs' % (end_time - start_time)))
        return data, train_result, predict_result
    elif cv_fold:
        print('parameters setting: cv fold = %f, aggregate = %s, threshold = %f, scale_indicator = %d, epochs = %d'
              % (cv_fold, aggregate, threshold, scale_indicator, epochs))
        accuracy_list = list()
        n_bags = len(bags)
        kf = cross_validation.KFold(n_bags, cv_fold, shuffle=False, random_state=0)
        cf = 1
        for train_idx, test_idx in kf:
            train_bag = list()
            train_label = list()
            for idx in train_idx:
                train_bag.append(copy.deepcopy(bags[idx]))
                train_label.append(bag_labels[idx])
            test_bag = list()
            test_label = list()
            for idx in test_idx:
                test_bag.append(copy.deepcopy(bags[idx]))
                test_label.append(bag_labels[idx])

            targets, scales, func_values = dd_classifier.train(train_bag, scale_indicator, epochs)
            p_bags_label, p_bags_prob, p_instances_label, p_instances_prob = dd_classifier.predict(targets, scales,
                                                                                                   func_values,
                                                                                                   test_bag,
                                                                                                   aggregate,
                                                                                                   threshold)
            accuracy = sum(test_label == p_bags_label) / len(test_bag)
            accuracy_list.append(accuracy)
            print('completed fold %d, accuracy is %f' % (cf, accuracy))
            cf += 1

        mean_accuracy = float(np.mean(np.asarray(accuracy_list)))
        print('mean accuracy with %d-fold cross validation is %f' % (cv_fold, mean_accuracy))
        end_time = time.clock()
        print(('The code for file ' + os.path.split(__file__)[1] + ' ran for %.1fs' % (end_time - start_time)))

        return accuracy_list
    else:
        pass


def miSVM_on_kaggle(split_ratio=None, cv_fold=None):
    from miSVM import MiSVM

    start_time = time.clock()
    classifier = MiSVM()
    target = 'Patient_1'
    bags, bag_labels = load_kaggle_data_into_bag(target)
    bags, bag_labels = data_preprocess_musk_svm(bags)
    if split_ratio is None and cv_fold is None:
        model = classifier.train(bags)
        p_bags_label = classifier.predict(bags, model)
        accuracy = sum(bag_labels == p_bags_label) / len(bags)
        print('training accuracy is: %f' % accuracy)
        end_time = time.clock()
        print(('The code for file ' + os.path.split(__file__)[1] + ' ran for %.1fs' % (end_time - start_time)))

    elif split_ratio:
        train_bag, test_bag, train_label, test_label = cross_validation.train_test_split(bags,
                                                                                         bag_labels,
                                                                                         test_size=split_ratio,
                                                                                         random_state=10)
        print(test_label)
        model, train_bag = classifier.train(train_bag)
        p_bags_label, p_bag_prob = classifier.predict(test_bag, model)
        accuracy = sum(test_label == p_bags_label) / len(test_bag)
        print('split ratio is %f, testing accuracy is %f' % (split_ratio, accuracy))
        end_time = time.clock()
        print(('The code for file ' + os.path.split(__file__)[1] + ' ran for %.1fs' % (end_time - start_time)))

        classifier.check_solution(train_bag)

        return p_bag_prob, test_label

    elif cv_fold:
        accuracy_list = list()
        n_bags = len(bags)
        kf = cross_validation.KFold(n_bags, cv_fold, shuffle=True, random_state=0)
        cf = 1
        for train_idx, test_idx in kf:
            train_bag = list()
            train_label = list()
            for idx in train_idx:
                train_bag.append(copy.deepcopy(bags[idx]))
                train_label.append(bag_labels[idx])
            test_bag = list()
            test_label = list()
            for idx in test_idx:
                test_bag.append(copy.deepcopy(bags[idx]))
                test_label.append(bag_labels[idx])

            model, train_bag = classifier.train(train_bag)
            classifier.check_solution(train_bag)
            p_bags_label, p_bags_prob = classifier.predict(test_bag, model)
            accuracy = sum(test_label == p_bags_label) / len(test_bag)
            accuracy_list.append(accuracy)
            print('completed fold %d, accuracy is %f' % (cf, accuracy))
            cf += 1

        mean_accuracy = float(np.mean(np.asarray(accuracy_list)))
        print('mean accuracy with %d-fold cross validation is %f' % (cv_fold, mean_accuracy))
        end_time = time.clock()
        print(('The code for file ' + os.path.split(__file__)[1] + ' ran for %.1fs' % (end_time - start_time)))

        return accuracy_list
    else:
        pass


if __name__ == '__main__':
    # data, _, predict_result = EMDD_inst_on_kaggle(split_ratio=None, cv_fold=10, aggregate='min', threshold=0.5, scale_indicator=1, epochs=5)
    # _, p_bags_prob, _, _ = predict_result
    # _, _, _, test_labels = data
    # plot_roc(test_labels, p_bags_prob)

    # EMDD_bag_on_kaggle(split_ratio=0.2, cv_fold=None, threshold=0.5, scale_indicator=1, epochs=2)
    # maxDD_on_kaggle(split_ratio=None, cv_fold=None, aggregate='min', threshold=0.5, scale_indicator=1, epochs=2)
    p_bags_prob, test_label = miSVM_on_kaggle(split_ratio=0.2, cv_fold=None)
    print(p_bags_prob)
    print(test_label)
    plot_roc(test_label, p_bags_prob)


    # miSVM_on_kaggle(split_ratio=None, cv_fold=10)

