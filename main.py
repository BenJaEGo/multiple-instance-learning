import time
from sklearn import cross_validation

import copy
import os
import numpy as np
import random
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from scipy import interp
from sklearn.preprocessing import label_binarize


from data_utils import load_musk_data
from preprocess_utils import data_preprocess_musk_dd
from preprocess_utils import data_preprocess_musk_svm




def maxDD_inst_method(split_ratio=None, cv_fold=None, aggregate='min', threshold=0.5, scale_indicator=1, epochs=10):
    from maxDD_inst import MaxDiverseDensity

    start_time = time.clock()
    dd_classifier = MaxDiverseDensity()

    file_path = 'musk1.txt'

    bags, bag_labels = load_musk1_data(file_path)
    bags = data_preprocess_musk_dd(bags)
    if split_ratio is None and cv_fold is None:
        print('parameters setting: split_ratio = None, cv_fold = None, aggregate = %s, threshold = %.4f, '
              'scale_indicator = %d, epochs = %d' % (aggregate, threshold, scale_indicator, epochs))
        targets, scales, nll_costs = dd_classifier.train(bags, scale_indicator, epochs)
        p_bags_label, p_bags_prob, p_instances_label, p_instances_prob = dd_classifier.predict(targets, scales,
                                                                                               nll_costs, bags,
                                                                                               aggregate, threshold)
        accuracy = sum(bag_labels == p_bags_label) / len(bags)
        print('training accuracy is: %f' % accuracy)
        end_time = time.clock()
        print(('The code for file ' + os.path.split(__file__)[1] + ' ran for %.1fs' % (end_time - start_time)))

        fpr, tpr, _ = roc_curve(bag_labels, p_bags_prob)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, 'k--', label='training (area = %0.2f)' % roc_auc)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('maxDD_inst, acc = %.3f' % accuracy)
        plt.legend(loc="lower right")
        plt.show()

    elif split_ratio:
        print('parameters setting: split ratio = %f, cv_fold = None, aggregate = %s, '
              'threshold = %.4f, scale_indicator = %d, epochs = %d' %
              (split_ratio, aggregate, threshold, scale_indicator, epochs))
        train_bag, test_bag, train_label, test_label = cross_validation.train_test_split(bags,
                                                                                         bag_labels,
                                                                                         test_size=split_ratio,
                                                                                         random_state=0)

        targets, scales, nll_costs = dd_classifier.train(train_bag, scale_indicator, epochs)
        p_bags_label, p_bags_prob, p_instances_label, p_instances_prob = dd_classifier.predict(targets, scales,
                                                                                               nll_costs, test_bag,
                                                                                               aggregate, threshold)
        accuracy = sum(test_label == p_bags_label) / len(test_bag)
        print('split ratio is %f, testing accuracy is %f' % (split_ratio, accuracy))
        end_time = time.clock()
        print(('The code for file ' + os.path.split(__file__)[1] + ' ran for %.1fs' % (end_time - start_time)))

        if np.any(np.asarray(test_label) == 1) and np.any(np.asarray(test_label) == 0):
            fpr, tpr, _ = roc_curve(test_label, p_bags_prob)
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, 'k--', label='ROC (area = %0.2f)' % roc_auc)
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('maxDD_inst, split ratio %.2f, acc %f ' % (split_ratio, accuracy))
            plt.legend(loc="lower right")
            plt.show()

    elif cv_fold:
        print('parameters setting: cv fold = %d, aggregate = %s, threshold = %.4f, scale_indicator = %d, epochs = %d'
              % (cv_fold, aggregate, threshold, scale_indicator, epochs))

        accuracy_list = list()
        n_bags = len(bags)
        random_seed = random.randint(1, 1000)
        print('random seed is:', random_seed)
        kf = cross_validation.KFold(n_bags, cv_fold, shuffle=True, random_state=random_seed)
        cf = 1

        mean_tpr = 0.0
        mean_fpr = np.linspace(0, 1, 100)
        n_roc = 0

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

            targets, scales, nll_costs = dd_classifier.train(train_bag, scale_indicator, epochs)
            p_bags_label, p_bags_prob, p_instances_label, p_instances_prob = dd_classifier.predict(targets, scales,
                                                                                                   nll_costs,
                                                                                                   test_bag,
                                                                                                   aggregate,
                                                                                                   threshold)
            accuracy = sum(np.asarray(test_label).squeeze() == p_bags_label) / len(test_bag)
            accuracy_list.append(accuracy)
            print('completed fold %d, accuracy is %f' % (cf, accuracy))
            print('test label: ', np.asarray(test_label).squeeze())
            print('predict label:', p_bags_label)
            print('predict probabilities: ', p_bags_prob)

            if np.any(np.asarray(test_label) == 1) and np.any(np.asarray(test_label) == 0):
                fpr, tpr, thresholds = roc_curve(test_label, p_bags_prob)
                mean_tpr += interp(mean_fpr, fpr, tpr)
                mean_tpr[0] = 0.0
                roc_auc = auc(fpr, tpr)
                plt.plot(fpr, tpr, lw=1, label='fold %d (area = %0.2f)' % (cf, roc_auc))
                n_roc += 1
            else:
                pass

            cf += 1

        plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')
        mean_tpr /= n_roc
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        plt.plot(mean_fpr, mean_tpr, 'k--',
                 label='Mean ROC (area = %0.2f)' % mean_auc, lw=2)

        mean_accuracy = float(np.mean(np.asarray(accuracy_list)))
        print('mean accuracy with %d-fold cross validation is %f' % (cv_fold, mean_accuracy))
        end_time = time.clock()
        print(('The code for file ' + os.path.split(__file__)[1] + ' ran for %.1fs' % (end_time - start_time)))

        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('maxDD_inst, acc = %.3f' % mean_accuracy)
        plt.legend(loc="lower right")
        plt.show()
    else:
        pass


def EMDD_inst_method(split_ratio=None, cv_fold=None, aggregate='min', threshold=0.5, scale_indicator=1, epochs=10):
    from EMDD_inst import EMDiverseDensity

    start_time = time.clock()
    dd_classifier = EMDiverseDensity()

    file_path = 'musk1.txt'

    bags, bag_labels = load_musk1_data(file_path)
    bags = data_preprocess_musk_dd(bags)
    if split_ratio is None and cv_fold is None:
        print('parameters setting: split_ratio = None, cv_fold = None, aggregate = %s, threshold = %.4f, '
              'scale_indicator = %d, epochs = %d' % (aggregate, threshold, scale_indicator, epochs))
        targets, scales, nll_costs = dd_classifier.train(bags, scale_indicator, epochs)
        p_bags_label, p_bags_prob, p_instances_label, p_instances_prob = dd_classifier.predict(targets, scales,
                                                                                               nll_costs, bags,
                                                                                               aggregate, threshold)
        accuracy = sum(bag_labels == p_bags_label) / len(bags)
        print('training accuracy is: %f' % accuracy)
        end_time = time.clock()
        print(('The code for file ' + os.path.split(__file__)[1] + ' ran for %.1fs' % (end_time - start_time)))

        fpr, tpr, _ = roc_curve(bag_labels, p_bags_prob)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, 'k--', label='training (area = %0.2f)' % roc_auc)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('EMDD_inst, acc = %.3f' % accuracy)
        plt.legend(loc="lower right")
        plt.show()

    elif split_ratio:
        print('parameters setting: split ratio = %f, cv_fold = None, aggregate = %s, '
              'threshold = %.4f, scale_indicator = %d, epochs = %d' %
              (split_ratio, aggregate, threshold, scale_indicator, epochs))
        train_bag, test_bag, train_label, test_label = cross_validation.train_test_split(bags,
                                                                                         bag_labels,
                                                                                         test_size=split_ratio,
                                                                                         random_state=0)

        targets, scales, nll_costs = dd_classifier.train(train_bag, scale_indicator, epochs)
        p_bags_label, p_bags_prob, p_instances_label, p_instances_prob = dd_classifier.predict(targets, scales,
                                                                                               nll_costs, test_bag,
                                                                                               aggregate, threshold)
        accuracy = sum(test_label == p_bags_label) / len(test_bag)
        print('split ratio is %f, testing accuracy is %f' % (split_ratio, accuracy))
        end_time = time.clock()
        print(('The code for file ' + os.path.split(__file__)[1] + ' ran for %.1fs' % (end_time - start_time)))

        if np.any(np.asarray(test_label) == 1) and np.any(np.asarray(test_label) == 0):
            fpr, tpr, _ = roc_curve(test_label, p_bags_prob)
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, 'k--', label='ROC (area = %0.2f)' % roc_auc)
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('EMDD_inst, split ratio %.2f, acc %f ' % (split_ratio, accuracy))
            plt.legend(loc="lower right")
            plt.show()

    elif cv_fold:
        print('parameters setting: cv fold = %d, aggregate = %s, threshold = %.4f, scale_indicator = %d, epochs = %d'
              % (cv_fold, aggregate, threshold, scale_indicator, epochs))

        accuracy_list = list()
        n_bags = len(bags)
        random_seed = random.randint(1, 1000)
        print('random seed is:', random_seed)
        kf = cross_validation.KFold(n_bags, cv_fold, shuffle=True, random_state=random_seed)
        cf = 1

        mean_tpr = 0.0
        mean_fpr = np.linspace(0, 1, 100)
        n_roc = 0

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

            targets, scales, nll_costs = dd_classifier.train(train_bag, scale_indicator, epochs)
            p_bags_label, p_bags_prob, p_instances_label, p_instances_prob = dd_classifier.predict(targets, scales,
                                                                                                   nll_costs,
                                                                                                   test_bag,
                                                                                                   aggregate,
                                                                                                   threshold)
            accuracy = sum(np.asarray(test_label).squeeze() == p_bags_label) / len(test_bag)
            accuracy_list.append(accuracy)
            print('completed fold %d, accuracy is %f' % (cf, accuracy))
            print('test label: ', np.asarray(test_label).squeeze())
            print('predict label:', p_bags_label)
            print('predict probabilities: ', p_bags_prob)

            if np.any(np.asarray(test_label) == 1) and np.any(np.asarray(test_label) == 0):
                fpr, tpr, thresholds = roc_curve(test_label, p_bags_prob)
                mean_tpr += interp(mean_fpr, fpr, tpr)
                mean_tpr[0] = 0.0
                roc_auc = auc(fpr, tpr)
                plt.plot(fpr, tpr, lw=1, label='fold %d (area = %0.2f)' % (cf, roc_auc))
                n_roc += 1
            else:
                pass

            cf += 1

        plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')
        mean_tpr /= n_roc
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        plt.plot(mean_fpr, mean_tpr, 'k--',
                 label='Mean ROC (area = %0.2f)' % mean_auc, lw=2)

        mean_accuracy = float(np.mean(np.asarray(accuracy_list)))
        print('mean accuracy with %d-fold cross validation is %f' % (cv_fold, mean_accuracy))
        end_time = time.clock()
        print(('The code for file ' + os.path.split(__file__)[1] + ' ran for %.1fs' % (end_time - start_time)))

        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('EMDD_inst, acc = %.3f' % mean_accuracy)
        plt.legend(loc="lower right")
        plt.show()
    else:
        pass


def EMDD_bag_method(split_ratio=None, cv_fold=None, threshold=0.5, scale_indicator=1, epochs=10):
    from EMDD_bag import EMDiverseDensity

    start_time = time.clock()
    dd_classifier = EMDiverseDensity()

    file_path = 'musk1.txt'

    bags, bag_labels = load_musk1_data(file_path)
    bags = data_preprocess_musk_dd(bags)
    if split_ratio is None and cv_fold is None:
        print('parameters setting: split_ratio = None, cv_fold = None, threshold = %.4f, '
              'scale_indicator = %d, epochs = %d' % (threshold, scale_indicator, epochs))
        target, scale = dd_classifier.train(bags, scale_indicator, epochs, threshold)
        p_bags_label, p_bags_prob, p_instances_label, p_instances_prob = dd_classifier.predict(target, scale,
                                                                                               bags, threshold)
        accuracy = sum(bag_labels == p_bags_label) / len(bags)
        print('training accuracy is: %f' % accuracy)
        end_time = time.clock()
        print(('The code for file ' + os.path.split(__file__)[1] + ' ran for %.1fs' % (end_time - start_time)))

        fpr, tpr, _ = roc_curve(bag_labels, p_bags_prob)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, 'k--', label='training (area = %0.2f)' % roc_auc)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('EMDD_bag, acc = %.3f' % accuracy)
        plt.legend(loc="lower right")
        plt.show()

    elif split_ratio:
        print('parameters setting: split ratio = %f, cv_fold = None, '
              'threshold = %.4f, scale_indicator = %d, epochs = %d' %
              (split_ratio, threshold, scale_indicator, epochs))
        train_bag, test_bag, train_label, test_label = cross_validation.train_test_split(bags,
                                                                                         bag_labels,
                                                                                         test_size=split_ratio,
                                                                                         random_state=0)

        target, scale = dd_classifier.train(train_bag, scale_indicator, epochs, threshold)
        p_bags_label, p_bags_prob, p_instances_label, p_instances_prob = dd_classifier.predict(target, scale,
                                                                                               test_bag,
                                                                                               threshold)
        accuracy = sum(test_label == p_bags_label) / len(test_bag)
        print('split ratio is %f, testing accuracy is %f' % (split_ratio, accuracy))
        end_time = time.clock()
        print(('The code for file ' + os.path.split(__file__)[1] + ' ran for %.1fs' % (end_time - start_time)))

        if np.any(np.asarray(test_label) == 1) and np.any(np.asarray(test_label) == 0):
            fpr, tpr, _ = roc_curve(test_label, p_bags_prob)
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, 'k--', label='ROC (area = %0.2f)' % roc_auc)
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('EMDD_bag, split ratio %.2f, acc %f ' % (split_ratio, accuracy))
            plt.legend(loc="lower right")
            plt.show()

    elif cv_fold:
        print('parameters setting: cv fold = %d, threshold = %.4f, scale_indicator = %d, epochs = %d'
              % (cv_fold, threshold, scale_indicator, epochs))

        accuracy_list = list()
        n_bags = len(bags)
        random_seed = random.randint(1, 1000)
        print('random seed is:', random_seed)
        kf = cross_validation.KFold(n_bags, cv_fold, shuffle=True, random_state=random_seed)
        cf = 1

        mean_tpr = 0.0
        mean_fpr = np.linspace(0, 1, 100)
        n_roc = 0

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

            target, scale = dd_classifier.train(train_bag, scale_indicator, epochs, threshold)
            p_bags_label, p_bags_prob, p_instances_label, p_instances_prob = dd_classifier.predict(target, scale,
                                                                                                   test_bag,
                                                                                                   threshold)
            accuracy = sum(np.asarray(test_label).squeeze() == p_bags_label) / len(test_bag)
            accuracy_list.append(accuracy)
            print('completed fold %d, accuracy is %f' % (cf, accuracy))
            print('test label: ', np.asarray(test_label).squeeze())
            print('predict label:', p_bags_label)
            print('predict probabilities: ', p_bags_prob)

            if np.any(np.asarray(test_label) == 1) and np.any(np.asarray(test_label) == 0):
                fpr, tpr, thresholds = roc_curve(test_label, p_bags_prob)
                mean_tpr += interp(mean_fpr, fpr, tpr)
                mean_tpr[0] = 0.0
                roc_auc = auc(fpr, tpr)
                plt.plot(fpr, tpr, lw=1, label='fold %d (area = %0.2f)' % (cf, roc_auc))
                n_roc += 1
            else:
                pass

            cf += 1

        plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')
        mean_tpr /= n_roc
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        plt.plot(mean_fpr, mean_tpr, 'k--',
                 label='Mean ROC (area = %0.2f)' % mean_auc, lw=2)

        mean_accuracy = float(np.mean(np.asarray(accuracy_list)))
        print('mean accuracy with %d-fold cross validation is %f' % (cv_fold, mean_accuracy))
        end_time = time.clock()
        print(('The code for file ' + os.path.split(__file__)[1] + ' ran for %.1fs' % (end_time - start_time)))

        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('EMDD_bag, acc = %.3f' % mean_accuracy)
        plt.legend(loc="lower right")
        plt.show()
    else:
        pass


def miSVM_inst_method(split_ratio=None, cv_fold=None):
    from MISVM_inst import MiSVM

    start_time = time.clock()
    classifier = MiSVM()

    file_path = 'musk1.txt'

    bags, bag_labels = load_musk_data(file_path)
    bags, bag_labels = data_preprocess_musk_svm(bags)

    if split_ratio is None and cv_fold is None:

        model, train_bag = classifier.train(bags)

        classifier.check_solution(train_bag)
        params = model.get_params()
        param_c = params['C']
        param_gamma = params['gamma']

        p_bags_label, p_bags_dist = classifier.predict(bags, model)
        accuracy = sum(bag_labels == p_bags_label) / len(bags)
        print('C = %.2f, gamma = %.2f, acc = %f' % (param_c, param_gamma, accuracy))
        print('test label: ', bag_labels)
        print('predict label:', p_bags_label)
        print('predict distance: ', p_bags_dist)
        end_time = time.clock()
        print(('The code for file ' + os.path.split(__file__)[1] + ' ran for %.1fs' % (end_time - start_time)))

        fpr, tpr, _ = roc_curve(bag_labels, p_bags_dist)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, 'k--', label='training (area = %0.2f)' % roc_auc)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('MISVM_inst, C = %.2f, gamma = %.2f, acc = %.3f' % (param_c, param_gamma, accuracy))
        plt.legend(loc="lower right")
        plt.show()

    elif split_ratio:
        random_seed = random.randint(1, 1000)
        print('random seed is: %d' % random_seed)
        train_bag, test_bag, train_label, test_label = cross_validation.train_test_split(bags,
                                                                                         bag_labels,
                                                                                         test_size=split_ratio,
                                                                                         random_state=random_seed)
        print('test positive bags number is %d' % np.sum(np.asarray(test_label) == 1))
        model, train_bag = classifier.train(train_bag)

        classifier.check_solution(train_bag)
        params = model.get_params()
        param_c = params['C']
        param_gamma = params['gamma']

        p_bags_label, p_bags_dist = classifier.predict(test_bag, model)
        accuracy = sum(test_label == p_bags_label) / len(test_bag)
        print('C = %.2f, gamma = %.2f, split ratio = %f, testing accuracy = %f' %
              (param_c, param_gamma, split_ratio, accuracy))
        print('test label: ', test_label)
        print('predict label:', p_bags_label)
        print('predict distance: ', p_bags_dist)
        end_time = time.clock()
        print(('The code for file ' + os.path.split(__file__)[1] + ' ran for %.1fs' % (end_time - start_time)))

        if np.any(np.asarray(test_label) == 1) and np.any(np.asarray(test_label) == 0):
            fpr, tpr, _ = roc_curve(test_label, p_bags_dist)
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, 'k--', label='ROC (area = %0.2f)' % roc_auc)
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('MISVM_inst C = %.2f, gamma = %.2f, split ratio = %.2f, acc = %f ' %
                      (param_c, param_gamma, split_ratio, accuracy))
            plt.legend(loc="lower right")
            plt.show()

    elif cv_fold:
        accuracy_list = list()
        n_bags = len(bags)
        random_seed = random.randint(1, 1000)
        print('random seed is:', random_seed)
        kf = cross_validation.KFold(n_bags, cv_fold, shuffle=True, random_state=random_seed)
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
            params = model.get_params()
            param_c = params['C']
            param_gamma = params['gamma']

            p_bags_label, p_bags_dist = classifier.predict(test_bag, model)
            accuracy = sum(np.asarray(test_label).squeeze() == p_bags_label) / len(test_bag)
            accuracy_list.append(accuracy)
            print('completed fold %d, accuracy is %f' % (cf, accuracy))
            print('test label: ', test_label)
            print('predict label:', p_bags_label)
            print('predict distance: ', p_bags_dist)

            if np.any(np.asarray(test_label) == 1) and np.any(np.asarray(test_label) == 0):
                fpr, tpr, thresholds = roc_curve(test_label, p_bags_dist)
                mean_tpr += interp(mean_fpr, fpr, tpr)
                mean_tpr[0] = 0.0
                roc_auc = auc(fpr, tpr)
                plt.plot(fpr, tpr, lw=1, label='fold %d (area = %0.2f)' % (cf, roc_auc))
                n_roc += 1
            else:
                pass

            cf += 1
            p_values.append(p_bags_dist)
            test_labels.append(test_label)

        plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')
        mean_tpr /= n_roc
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        plt.plot(mean_fpr, mean_tpr, 'k--',
                 label='Mean ROC (area = %0.2f)' % mean_auc, lw=2)

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


def miSVM_bag_method(split_ratio=None, cv_fold=None):
    from MISVM_bag import MiSVM

    start_time = time.clock()
    classifier = MiSVM()

    file_path = 'musk1.txt'

    bags, bag_labels = load_musk_data(file_path)
    bags, bag_labels = data_preprocess_musk_svm(bags)

    if split_ratio is None and cv_fold is None:

        model, train_bag = classifier.train(bags)

        classifier.check_solution(train_bag)
        params = model.get_params()
        param_c = params['C']
        param_gamma = params['gamma']

        p_bags_label, p_bags_dist = classifier.predict(bags, model)
        accuracy = sum(bag_labels == p_bags_label) / len(bags)
        print('C = %.2f, gamma = %.2f, acc = %f' % (param_c, param_gamma, accuracy))
        print('test label: ', bag_labels)
        print('predict label:', p_bags_label)
        print('predict distance: ', p_bags_dist)
        end_time = time.clock()
        print(('The code for file ' + os.path.split(__file__)[1] + ' ran for %.1fs' % (end_time - start_time)))

        fpr, tpr, _ = roc_curve(bag_labels, p_bags_dist)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, 'k--', label='training (area = %0.2f)' % roc_auc)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('MISVM_bag, C = %.2f, gamma = %.2f, acc = %.3f' % (param_c, param_gamma, accuracy))
        plt.legend(loc="lower right")
        plt.show()

    elif split_ratio:
        random_seed = random.randint(1, 1000)
        print('random seed is: %d' % random_seed)
        train_bag, test_bag, train_label, test_label = cross_validation.train_test_split(bags,
                                                                                         bag_labels,
                                                                                         test_size=split_ratio,
                                                                                         random_state=random_seed)
        print('test positive bags number is %d' % np.sum(np.asarray(test_label) == 1))
        model, train_bag = classifier.train(train_bag)

        classifier.check_solution(train_bag)
        params = model.get_params()
        param_c = params['C']
        param_gamma = params['gamma']

        p_bags_label, p_bags_dist = classifier.predict(test_bag, model)
        accuracy = sum(test_label == p_bags_label) / len(test_bag)
        print('C = %.2f, gamma = %.2f, split ratio = %f, testing accuracy = %f' %
              (param_c, param_gamma, split_ratio, accuracy))
        print('test label: ', test_label)
        print('predict label:', p_bags_label)
        print('predict distance: ', p_bags_dist)
        end_time = time.clock()
        print(('The code for file ' + os.path.split(__file__)[1] + ' ran for %.1fs' % (end_time - start_time)))

        if np.any(np.asarray(test_label) == 1) and np.any(np.asarray(test_label) == 0):
            fpr, tpr, _ = roc_curve(test_label, p_bags_dist)
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, 'k--', label='ROC (area = %0.2f)' % roc_auc)
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('MISVM_bag C = %.2f, gamma = %.2f, split ratio = %.2f, acc = %f ' %
                      (param_c, param_gamma, split_ratio, accuracy))
            plt.legend(loc="lower right")
            plt.show()

    elif cv_fold:
        accuracy_list = list()
        n_bags = len(bags)
        random_seed = random.randint(1, 1000)
        print('random seed is:', random_seed)
        kf = cross_validation.KFold(n_bags, cv_fold, shuffle=True, random_state=random_seed)
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
            params = model.get_params()
            param_c = params['C']
            param_gamma = params['gamma']

            p_bags_label, p_bags_dist = classifier.predict(test_bag, model)
            accuracy = sum(np.asarray(test_label).squeeze() == p_bags_label) / len(test_bag)
            accuracy_list.append(accuracy)
            print('completed fold %d, accuracy is %f' % (cf, accuracy))
            print('test label: ', test_label)
            print('predict label:', p_bags_label)
            print('predict distance: ', p_bags_dist)

            if np.any(np.asarray(test_label) == 1) and np.any(np.asarray(test_label) == 0):
                fpr, tpr, thresholds = roc_curve(test_label, p_bags_dist)
                mean_tpr += interp(mean_fpr, fpr, tpr)
                mean_tpr[0] = 0.0
                roc_auc = auc(fpr, tpr)
                plt.plot(fpr, tpr, lw=1, label='fold %d (area = %0.2f)' % (cf, roc_auc))
                n_roc += 1
            else:
                pass

            cf += 1
            p_values.append(p_bags_dist)
            test_labels.append(test_label)

        plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')
        mean_tpr /= n_roc
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        plt.plot(mean_fpr, mean_tpr, 'k--',
                 label='Mean ROC (area = %0.2f)' % mean_auc, lw=2)

        mean_accuracy = float(np.mean(np.asarray(accuracy_list)))
        print('mean accuracy with %d-fold cross validation is %f' % (cv_fold, mean_accuracy))
        end_time = time.clock()
        print(('The code for file ' + os.path.split(__file__)[1] + ' ran for %.1fs' % (end_time - start_time)))

        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('MISVM_bag, C = %.2f, gamma = %.2f, acc = %.3f' % (param_c, param_gamma, mean_accuracy))
        plt.legend(loc="lower right")
        plt.show()

        return p_values, test_labels
    else:
        pass


if __name__ == '__main__':

    # maxDD_inst_method(split_ratio=None, cv_fold=10, aggregate='min', threshold=0.5, scale_indicator=1, epochs=1)

    # EMDD_inst_method(split_ratio=None, cv_fold=10, aggregate='min', threshold=0.5, scale_indicator=1, epochs=1)
    # EMDD_bag_method(split_ratio=None, cv_fold=10, threshold=0.5, scale_indicator=1, epochs=3)

    # miSVM_inst_method()
    # miSVM_inst_method(split_ratio=0.2)
    # miSVM_inst_method(cv_fold=10)

    # miSVM_bag_method()
    # miSVM_bag_method(split_ratio=0.2)
    miSVM_bag_method(cv_fold=10)
