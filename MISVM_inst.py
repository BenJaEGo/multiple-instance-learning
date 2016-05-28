import numpy as np
from sklearn import svm
import time

_floatX = np.float32
_intX = np.int8


class MiSVM(object):

    def __init__(self):
        pass

    def collect_insts_labels(self, bags):
        instances = list()
        inst_labels = list()
        for bag in bags:
            instances.extend(bag['instances'])
            inst_labels.extend(bag['inst_labels'])
        instances = np.asarray(instances)
        inst_labels = np.asarray(inst_labels)
        return instances, inst_labels

    def train(self, bags):

        n_iter = 0
        x_train, y_train = self.collect_insts_labels(bags)
        pre_y_train = y_train

        clf = svm.SVC(kernel='rbf', C=1.0, gamma=0.1, probability=True, decision_function_shape='ovr')
        print("iter %d, Fitting the classifier to the training set, " % n_iter)
        t0 = time.time()
        clf.fit(x_train, y_train)
        print("iter %d done in %0.3fs" % (n_iter, (time.time() - t0)))

        while True:
            n_iter += 1
            for bag in bags:
                if 1 == bag['label']:

                    dist = clf.decision_function(bag['instances'])
                    p_label = clf.predict(bag['instances'])

                    if np.any(np.asarray(p_label) == 1):
                        bag['inst_labels'] = p_label
                    else:
                        max_idx = np.argmax(dist)
                        n_instances = bag['inst_labels'].shape
                        bag['inst_labels'] = np.ones(n_instances, ) * -1
                        bag['inst_labels'][max_idx] = 1

                elif -1 == bag['label']:
                    n_instances = bag['inst_labels'].shape
                    bag['inst_labels'] = np.ones(n_instances,) * -1
                else:
                    raise NotImplementedError('more than two classes.')

            x_train, y_train = self.collect_insts_labels(bags)
            print("iter %d, Fitting the classifier to the training set " % n_iter, end='')
            t0 = time.time()
            clf.fit(x_train, y_train)
            print("done in %0.3fs" % (time.time() - t0))

            y_diff = y_train - pre_y_train
            pre_y_train = y_train

            if np.sum(y_diff) == 0:
                print('iter %d done, label difference %d, break' % (n_iter, abs(np.sum(y_diff))))
                break

            print('iter %d done, label difference %d' % (n_iter, abs(np.sum(y_diff))))

        return clf, bags

    def predict(self, bags, clf):
        p_bag_labels = list()
        p_bag_prob = list()
        for bag in bags:
            x = bag['instances']
            y = bag['inst_labels']
            p_labels = clf.predict(x)
            bag_label = np.max(p_labels)
            p_bag_labels.append(bag_label)
            # negative label (-1) appears before positive label (1)
            p_bag_prob.append(np.max(clf.predict_proba(x)[:, 1]))

        p_bag_labels = np.asarray(p_bag_labels)
        p_bag_prob = np.asarray(p_bag_prob)
        return p_bag_labels, p_bag_prob

    def check_solution(self, bags):
        for bag in bags:
            if bag['label'] == 1:
                if np.any(np.asarray(bag['inst_labels']) == 1):
                    pass
                else:
                    raise RuntimeError('solution check failed for positive bags..something wrong happened..')
        print('positive bags check ends.')
        for bag in bags:
            if bag['label'] == -1:
                if np.all(np.asarray(bag['inst_labels']) == -1):
                    pass
                else:

                    raise RuntimeError('solution check failed for negative bags..something wrong happened..')
        print('negative bags check ends.')


