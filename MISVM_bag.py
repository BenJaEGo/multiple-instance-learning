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
            if bag['label'] == 0:
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
            if bag['label'] == 0:
                instances.extend(bag['instances'])
                inst_labels.extend(bag['inst_labels'])
            elif bag['label'] == 1:
                _, n_dim = bag['instances'].shape
                max_dist_inst = bag['instances'][selector[max_idx]].reshape([1, n_dim])
                instances.extend(max_dist_inst)
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
        clf = svm.SVC(kernel='rbf', C=100, gamma=0.1, probability=True, decision_function_shape='ovr')
        print("iter %d, Fitting the classifier to the training set" % n_iter)
        t0 = time.time()
        clf.fit(x_train, y_train)
        print("iter %d done in %0.3fs" % (n_iter, (time.time() - t0)))

        selector = self.calc_selector(bags, clf)
        x_train, y_train = self.collect_insts_labels(bags, selector)
        prev_selector = selector

        while True:
            n_iter += 1
            print("iter %d, Fitting the classifier to the training set " % n_iter)
            t0 = time.time()
            clf.fit(x_train, y_train)
            print("iter %d done in %0.3fs" % (n_iter, (time.time() - t0)))

            selector = self.calc_selector(bags, clf)
            x_train, y_train = self.collect_insts_labels(bags, selector)
            selector_diff = prev_selector - selector
            prev_selector = selector

            if np.sum(selector_diff) == 0:
                print('iter %d done, selector difference is %d, break' % (n_iter, abs(np.sum(selector_diff))))
                break
            else:
                print('iter %d done, selector difference is %d' % (n_iter, abs(np.sum(selector_diff))))

        return clf, bags

    def predict(self, bags, clf):
        p_bags_label = list()
        p_bags_dist = list()
        for bag in bags:
            x = bag['instances']
            p_inst_labels = clf.predict(x)
            p_bag_label = np.max(p_inst_labels)

            p_inst_dist = clf.decision_function(x)

            p_bag_dist = np.max(p_inst_dist)

            p_bags_label.append(p_bag_label)
            p_bags_dist.append(p_bag_dist)

        p_bags_label = np.asarray(p_bags_label).squeeze()
        p_bags_dist = np.asarray(p_bags_dist).squeeze()
        return p_bags_label, p_bags_dist

    def check_solution(self, bags):
        for bag in bags:
            if bag['label'] == 1:
                if np.any(np.asarray(bag['inst_labels']) == 1):
                    pass
                else:
                    raise RuntimeError('solution check failed for positive bags..something wrong happened..')
        print('positive bags check ends.')
        for bag in bags:
            if bag['label'] == 0:
                if np.all(np.asarray(bag['inst_labels']) == 0):
                    pass
                else:
                    raise RuntimeError('solution check failed for negative bags..something wrong happened..')
        print('negative bags check ends.')
