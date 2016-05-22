
import pickle
import gzip
import time
from svmutil import *
from sklearn import svm

from data_utils import load_kaggle_data_into_instance, load_feature_from_txt
from vis_utils import plot_roc


def svm_lib(prob, x_test=None, y_test=None, target=None, params=None):

    start_time = time.clock()

    train_model = svm_train(prob, params)
    end_time = time.clock()
    print('time elapsed in training SVM: %f' % (end_time - start_time))
    # print(sum(y_test))
    if x_test is not None and y_test is not None:
        p_labels, p_acc, p_vals = svm_predict(y_test, x_test, train_model)

        # result = (p_labels, p_acc, p_vals, y_test)
        # filename = '%s_libsvm_params_%s_result.pkl.gz' % (target, params)
        # with gzip.open(filename, 'wb') as f:
        #     pickle.dump(result, f)
        return p_vals, p_acc


def svm_sklearn(x_train, y_train, x_test=None, y_test=None):
    clf = svm.SVC(kernel='rbf', C=100, gamma=0.1)
    clf.fit(x_train, y_train)
    if x_test is None and y_test is None:
        pred_label = clf.predict(x_train)
        # print(sum(pred_label))
        print(sum(pred_label == y_train) / len(y_train))
    else:
        pred_label = clf.predict(x_test)
        print(sum(pred_label == y_test) / len(y_test))


if __name__ == '__main__':

    target = 'Dog_1'

    # libsvm
    # x_train, x_test, y_train, y_test = load_kaggle_data_into_instance(target, cv_ratio=0.2)
    # print(x_train[:, 0])
    # x_train, y_train, x_test, y_test = load_kaggle_data_into_instance(target, cv_ratio=None)

    file_path = 'Dog_1_instance_5s_24_24_256d_normalize.txt'
    # x_train, y_train = load_feature_from_txt(file_path, cv_ratio=None)
    x_train, y_train, x_test, y_test = load_feature_from_txt(file_path, cv_ratio=0.2)
    # print(x_train[0, :])

    # x_train, y_train, x_test, y_test = x_train[1:10000], y_train[1:10000], x_test[1:1000], y_test[1:1000]
    # prob = svm_problem(y_train.tolist(), x_train.tolist())
    # for c in [0.001, 0.01, 0.1, 1, 10, 100, 1000]:
    #     for g in [0.001, 0.01, 0.1, 1, 10, 100, 1000]:
    #         # print(type(c))
    #         param = '-s 0 -t 2 -c ' + str(c) + ' -g ' + str(g)
    #         p_vals, p_acc = svm_lib(prob, x_test.tolist(), y_test.tolist(), target, param)
    #         # print(p_acc)
    #         print('c %f g %f, accuracy is: %f' % (c, g, p_acc[0]))
    #         plot_roc(y_test, p_vals)

    param = '-s 0 -t 2 -c 10 -g 0.1'
    prob = svm_problem(y_train.tolist(), x_train.tolist())
    p_vals, p_acc = svm_lib(prob, x_test.tolist(), y_test.tolist(), target, param)
    plot_roc(y_test, p_vals)

    # svm_sklearn(x_train, y_train)
    # svm_sklearn(x_train, y_train, x_test, y_test)
