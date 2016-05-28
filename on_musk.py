import time
from sklearn import cross_validation
from sklearn import preprocessing
import copy
import os
import numpy as np
import random
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from scipy import interp

from data_utils import load_musk1_data


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
            bag['label'] = -1
            bag['inst_labels'] = np.ones([n_instances, ]) * -1
            bag_labels.append(-1)
        else:
            bag['label'] = 1
            bag['inst_labels'] = np.ones([n_instances, ])
            bag_labels.append(1)
        instances.extend(bag['instances'])
        # bag['instances'] /= 100
        # bag['instances'] = preprocessing.minmax_scale(bag['instances'], axis=1, feature_range=(-1, 1))
        # bag['instances'] = preprocessing.normalize(bag['instances'], axis=1)
    instances = np.asarray(instances)
    # print(instances.shape)
    instances = preprocessing.minmax_scale(instances, axis=1, feature_range=(-1, 1))
    # instances = preprocessing.minmax_scale(instances, axis=1, feature_range=(0, 1))
    # instances = preprocessing.normalize(instances, norm='l2', axis=1)
    inst_idx = 0
    for bag in bags:
        [n_instances, _] = bag['instances'].shape
        bag['instances'] = instances[inst_idx: inst_idx + n_instances, :]
        inst_idx += n_instances
    return bags, bag_labels


def EMDD_inst_on_musk1(split_ratio=None, cv_fold=None,
                       aggregate='min', threshold=0.5, scale_indicator=1, epochs=10):
    from EMDD_inst import EMDiverseDensity
    start_time = time.clock()
    dd_classifier = EMDiverseDensity()
    file_path = 'musk1.txt'
    bags, bag_labels = load_musk1_data(file_path)
    bags = data_preprocess_musk_dd(bags)
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


def EMDD_bag_on_musk1(split_ratio=None, cv_fold=None, threshold=0.5, scale_indicator=1, epochs=10):
    from EMDD_bag import EMDiverseDensity
    start_time = time.clock()
    dd_classifier = EMDiverseDensity()
    file_path = 'musk1.txt'
    bags, bag_labels = load_musk1_data(file_path)
    bags = data_preprocess_musk_dd(bags)
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


def maxDD_on_musk1(split_ratio=None, cv_fold=None, aggregate='min', threshold=0.5, scale_indicator=1, epochs=10):
    from maxDD_inst import MaxDiverseDensity

    start_time = time.clock()
    dd_classifier = MaxDiverseDensity()
    file_path = 'musk1.txt'
    bags, bag_labels = load_musk1_data(file_path)
    bags = data_preprocess_musk_dd(bags)
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


def miSVM_on_musk1(split_ratio=None, cv_fold=None):
    from MISVM_bag import MiSVM

    start_time = time.clock()
    classifier = MiSVM()
    file_path = 'musk2.txt'
    bags, bag_labels = load_musk1_data(file_path)
    bags, bag_labels = data_preprocess_musk_svm(bags)
    if split_ratio is None and cv_fold is None:
        model, train_bag = classifier.train(bags)

        classifier.check_solution(train_bag)
        params = model.get_params()
        param_c = params['C']
        param_gamma = params['gamma']

        p_bags_label, p_bags_prob = classifier.predict(bags, model)
        print('test label: ', bag_labels)
        print('predict label: ', p_bags_prob)
        accuracy = sum(bag_labels == p_bags_label) / len(bags)
        print('training accuracy is: %f' % accuracy)
        end_time = time.clock()
        print(('The code for file ' + os.path.split(__file__)[1] + ' ran for %.1fs' % (end_time - start_time)))

        fpr, tpr, _ = roc_curve(bag_labels, p_bags_prob)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, 'k--', label='ROC training (area = %0.2f)' % roc_auc)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('MISVM_bag, C = %.2f, gamma = %.2f, acc = %.3f' % (param_c, param_gamma, accuracy))
        plt.legend(loc="lower right")
        plt.show()

        return p_bags_prob, bag_labels

    elif split_ratio:
        random_seed = random.randint(1, 1000)
        # random_seed = 712
        train_bag, test_bag, train_label, test_label = cross_validation.train_test_split(bags,
                                                                                         bag_labels,
                                                                                         test_size=split_ratio,
                                                                                         random_state=random_seed)
        print('positive bags number is %d' % np.sum(np.asarray(test_label) == 1))
        model, train_bag = classifier.train(train_bag)
        classifier.check_solution(train_bag)
        print(model.get_params())
        p_bags_label, p_bags_prob = classifier.predict(test_bag, model)
        accuracy = sum(test_label == p_bags_label) / len(test_bag)
        print('split ratio is %f, testing accuracy is %f' % (split_ratio, accuracy))
        end_time = time.clock()
        print(('The code for file ' + os.path.split(__file__)[1] + ' ran for %.1fs' % (end_time - start_time)))

        if np.any(np.asarray(test_label) == 1) and np.any(np.asarray(test_label) == -1):
            fpr, tpr, _ = roc_curve(test_label, p_bags_prob)
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, 'k--', label='ROC (area = %0.2f)' % roc_auc)
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('MISVM_INST split ratio %.2f acc %f ' % (split_ratio, accuracy))
            plt.legend(loc="lower right")
            plt.show()

    elif cv_fold:
        accuracy_list = list()
        n_bags = len(bags)
        kf = cross_validation.KFold(n_bags, cv_fold, shuffle=True, random_state=100)
        cf = 1
        p_values = list()
        test_labels = list()

        mean_tpr = 0.0
        mean_fpr = np.linspace(0, 1, 100)
        n_roc = 0

        param_c = 0
        param_gamma = 0

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
            # print(model.get_params())
            params = model.get_params()
            param_c = params['C']
            param_gamma = params['gamma']
            p_bags_label, p_bags_prob = classifier.predict(test_bag, model)
            accuracy = sum(test_label == p_bags_label) / len(test_bag)
            accuracy_list.append(accuracy)
            print('completed fold %d, accuracy is %f' % (cf, accuracy))

            if np.any(np.asarray(test_label) == 1) and np.any(np.asarray(test_label) == -1):
                fpr, tpr, thresholds = roc_curve(test_label, p_bags_prob)
                mean_tpr += interp(mean_fpr, fpr, tpr)
                mean_tpr[0] = 0.0
                roc_auc = auc(fpr, tpr)
                plt.plot(fpr, tpr, lw=1, label='fold %d (area = %0.2f)' % (cf, roc_auc))
                n_roc += 1
            else:
                pass

            cf += 1
            p_values.append(p_bags_prob)
            test_labels.append(test_label)

        plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')
        mean_tpr /= n_roc
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        plt.plot(mean_fpr, mean_tpr, 'k--', label='Mean ROC (area = %0.2f)' % mean_auc, lw=2)

        mean_accuracy = float(np.mean(np.asarray(accuracy_list)))
        print('mean accuracy with %d-fold cross validation is %f' % (cv_fold, mean_accuracy))
        end_time = time.clock()
        print(('The code for file ' + os.path.split(__file__)[1] + ' ran for %.1fs' % (end_time - start_time)))

        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('MISVM_inst, C = %.2f, gamma = %.2f, acc = %.3f' % (param_c, param_gamma, mean_accuracy))
        plt.legend(loc="lower right")
        plt.show()

        return p_values, test_labels
    else:
        pass

if __name__ == '__main__':
    # EMDD_inst_on_musk1(split_ratio=None, cv_fold=None, aggregate='min', threshold=0.5, scale_indicator=1, epochs=5)
    # EMDD_bag_on_musk1(split_ratio=None, cv_fold=None, threshold=0.5, scale_indicator=1, epochs=5)
    # maxDD_on_musk1(split_ratio=None, cv_fold=None, aggregate='min', threshold=0.5, scale_indicator=1, epochs=5)
    # miSVM_on_musk1()
    # miSVM_on_musk1(split_ratio=0.2)
    # miSVM_on_musk1(cv_fold=10)
    miSVM_on_musk1()



