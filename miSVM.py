import numpy as np
from sklearn import svm
from sklearn.grid_search import GridSearchCV
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

        x_train, y_train = self.collect_insts_labels(bags)
        pre_y_train = y_train

        clf = svm.SVC(kernel='rbf', C=100, gamma=0.01, probability=True, decision_function_shape='ovr')
        clf.fit(x_train, y_train)


        # print("Fitting the classifier to the training set")
        # t0 = time.time()
        # param_grid = {'C': [1e-1, 1e0, 1e1, 1e2, 1e3],
        #               'gamma': [0.001, 0.01, 0.1], }
        # clf = GridSearchCV(svm.SVC(kernel='rbf', class_weight='balanced', probability=True, decision_function_shape='ovr'), param_grid)
        # clf.fit(x_train, y_train)
        # print("done in %0.3fs" % (time.time() - t0))
        # print("Best estimator found by grid search:")
        # print(clf.best_estimator_)

        n_iter = 0

        while True:

            for bag in bags:
                if 1 == bag['label']:

                    dist = clf.decision_function(bag['instances'])
                    # print(dist)
                    p_label = clf.predict(bag['instances'])

                    if np.any(np.asarray(p_label) == 1):
                        bag['inst_labels'] = p_label
                    else:
                        max_idx = np.argmax(abs(dist))
                        n_instances = bag['inst_labels'].shape
                        bag['inst_labels'] = np.ones(n_instances, ) * -1
                        bag['inst_labels'][max_idx] = 1

                elif -1 == bag['label']:
                    n_instances = bag['inst_labels'].shape
                    bag['inst_labels'] = np.ones(n_instances,) * -1
                else:
                    raise NotImplementedError('more than two classes.')

            x_train, y_train = self.collect_insts_labels(bags)
            clf.fit(x_train, y_train)
            n_iter += 1
            y_diff = y_train - pre_y_train
            pre_y_train = y_train

            if np.sum(y_diff) == 0:
                print('iter %d difference %d, break' % (n_iter, abs(np.sum(y_diff))))
                break

            print('iter %d, difference %d' % (n_iter, abs(np.sum(y_diff))))

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
                    # print(bag['inst_labels'])
                else:
                    print('positive solution is bad, check your implementation.')
                    break
        print('positive bags check ends.')
        for bag in bags:
            if bag['label'] == -1:
                if np.all(np.asarray(bag['inst_labels']) == -1):
                    pass
                    # print(bag['inst_labels'])
                else:
                    print('negative solution is bad, check your implementation.')
                    break
        print('negative bags check ends.')

