import numpy as np
from sklearn import svm
import time

_floatX = np.float32
_intX = np.int8


class MiSVM(object):

    def __init__(self):
        pass

    def collect_initial_insts_labels(self, bags):
        instances = list()
        inst_labels = list()

        for bag in bags:
            if bag['label'] == -1:
                instances.extend(bag['instances'])
                inst_labels.extend(bag['inst_labels'])
            elif bag['label'] == 1:
                _, n_dim = bag['instances'].shape
                avg_inst = np.mean(bag['instances'], axis=0).reshape([1, n_dim])
                instances.extend(avg_inst)
                inst_labels.append(bag['label'])
            else:
                raise TypeError('incorrect instance label')
        instances = np.asarray(instances)
        inst_labels = np.asarray(inst_labels)

        return instances, inst_labels

    def collect_insts_labels(self, bags, selector):
        instances = list()
        inst_labels = list()

        max_idx = 0
        for bag in bags:
            if bag['label'] == -1:
                instances.extend(bag['instances'])
                inst_labels.extend(bag['inst_labels'])
            elif bag['label'] == 1:
                _, n_dim = bag['instances'].shape
                max_prob_inst = bag['instances'][selector[max_idx]].reshape([1, n_dim])
                instances.extend(max_prob_inst)
                inst_labels.append(bag['label'])
                max_idx += 1
            else:
                raise TypeError('incorrect instance label')
        instances = np.asarray(instances)
        inst_labels = np.asarray(inst_labels)

        return instances, inst_labels

    def calc_selector(self, bags, clf):
        selector = list()
        for bag in bags:
            if 1 == bag['label']:
                dist = clf.decision_function(bag['instances'])
                max_idx = int(np.argmax(dist))
                selector.append(max_idx)
        selector = np.asarray(selector)
        return selector

    def train(self, bags):
        n_iter = 0

        x_train, y_train = self.collect_initial_insts_labels(bags)
        clf = svm.SVC(kernel='rbf', C=1, gamma=0.1, probability=True, decision_function_shape='ovr')
        print("iter %d, Fitting the classifier to the training set, " % n_iter, end='')
        t0 = time.time()
        clf.fit(x_train, y_train)
        print("done in %0.3fs" % (time.time() - t0))

        selector = self.calc_selector(bags, clf)
        x_train, y_train = self.collect_insts_labels(bags, selector)
        past_selector_idx = selector

        while True:
            n_iter += 1
            print("iter %d, Fitting the classifier to the training set " % n_iter)
            t0 = time.time()
            clf.fit(x_train, y_train)
            print("iter %d done in %0.3fs" % (n_iter, (time.time() - t0)))

            selector = self.calc_selector(bags, clf)
            x_train, y_train = self.collect_insts_labels(bags, selector)
            selector_diff = past_selector_idx - selector
            past_selector_idx = selector

            if np.sum(selector_diff) == 0:
                print('iter %d done, selector difference %d, break' % (n_iter, abs(np.sum(selector_diff))))
                break

            print('iter %d done, selector difference %d' % (n_iter, abs(np.sum(selector_diff))))

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
