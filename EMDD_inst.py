import numpy as np
from scipy import optimize
import random


_floatX = np.float32
_intX = np.int8


class EMDiverseDensity(object):
    """
        bags is a list of bag
        each bag is a dict required following <key, value>
        key: inst_prob, value: a vector indicating each instance's probability
        key: label, value: a scalar indicating this bag's label
        key: prob, value: a scalar indicating this bag's probability
        key: instances, value: a numpy array indicating instances in this bag, each row is a instance, each column is a
        feature
        this version select given number of different positive instances in different bags as starting points for em
        in predict process, simply use 'min', 'max' or 'avg' mode to select a concept from these learned concepts using
        the negative log likelihood, then use the concept for testing data.
    """

    def __init__(self):
        pass

    def diverse_density_nll(self, params, instances, labels):
        [n_instances, n_dim] = instances.shape
        if params.shape[0] == n_dim:
            target = params
            scale = np.ones(n_dim, )
        else:
            target = params[0:n_dim]
            scale = params[n_dim:]

        nll_cost = 0

        dist = np.mean(((instances - target) ** 2) * (scale ** 2), axis=1)
        inst_prob = np.exp(-dist)
        for inst_idx in range(n_instances):
            if labels[inst_idx] == 1:
                if inst_prob[inst_idx] == 0:
                    inst_prob[inst_idx] = 1e-10
                nll_cost += -np.log(inst_prob[inst_idx])
            else:
                if inst_prob[inst_idx] == 1:
                    inst_prob[inst_idx] = 1 - 1e-10
                nll_cost += -np.log(1 - inst_prob[inst_idx])
        return nll_cost

    def em(self, bags, scale_indicator, init_target, init_scale, tol=1e-5):
        target = init_target
        scale = init_scale

        # select an optimal instance from each bag according to current target and scale
        diff = np.inf
        prev_nll_cost = np.inf
        nll_cost = 0

        init_nll_cost = 0
        init_nll_cost_indicator = 1

        em_loop_count = 0
        while diff > tol:

            em_loop_count += 1
            if em_loop_count > 1000:
                raise NotImplementedError('em loop error, loop number is %d larger than 1000.' % em_loop_count)

            selected_instances = list()
            selected_labels = list()
            # select an instance with highest probability from each bag
            for bag in bags:
                instances = np.asarray(bag['instances'])
                [_, n_dim] = instances.shape
                dist = np.mean(((instances - target) ** 2) * (scale ** 2), axis=1)
                bag['inst_prob'] = np.exp(-dist)
                max_idx = np.argmax(bag['inst_prob'])
                selected_instances.append(instances[max_idx, :])
                selected_labels.append(bag['label'])

            selected_instances = np.asarray(selected_instances)

            if scale_indicator == 1:
                init_params = np.hstack((target, scale))
                if init_nll_cost_indicator == 1:
                    init_nll_cost = self.diverse_density_nll(init_params, selected_instances, selected_labels)
                    init_nll_cost_indicator = 0
                optimized_params = optimize.minimize(self.diverse_density_nll, init_params,
                                                     args=(selected_instances, selected_labels,), method='L-BFGS-B')
                target = optimized_params.x[0:n_dim]
                scale = optimized_params.x[n_dim:]
            else:
                init_params = target
                if init_nll_cost_indicator == 1:
                    init_nll_cost = self.diverse_density_nll(init_params, selected_instances, selected_labels)
                    init_nll_cost_indicator = 0
                optimized_params = optimize.minimize(self.diverse_density_nll, init_params,
                                                     args=(selected_instances, selected_labels,), method='L-BFGS-B')
                target = optimized_params.x
                scale = np.ones(n_dim,)

            nll_cost = optimized_params.fun
            diff = prev_nll_cost - nll_cost
            prev_nll_cost = nll_cost

        print('em phase completed, loop number is %d. ' % em_loop_count, end='')

        return target, scale, nll_cost, init_nll_cost

    def train(self, bags, scale_indicator, epochs):
        n_bag = len(bags)
        n_pos_bag = 0

        n_pos_instances = 0
        for bag in bags:
            if bag['label'] == 1:
                n_pos_bag += 1
                n_pos_instances += bag['instances'].shape[0]

        epochs = min(n_pos_instances, epochs)
        print('training, total epochs number is %d, #positive bags is %d, #positive instances is %d'
              % (epochs, n_pos_bag, n_pos_instances))

        targets = list()
        scales = list()
        nll_costs = list()

        for epoch_idx in range(epochs):
            # randomly select a positive bag
            bag_idx = random.randint(0, n_bag - 1)
            while bags[bag_idx]['label'] == 0 or np.all(np.asarray(bags[bag_idx]['starting_point']) == 1):
                bag_idx = random.randint(0, n_bag - 1)
            # randomly select a positive instance not used before
            [_, n_dim] = bags[bag_idx]['instances'].shape
            starting_points = np.asarray(bags[bag_idx]['starting_point'])
            valuable_starting_points = np.flatnonzero(starting_points == 0)
            if valuable_starting_points.shape[0] == 1:
                instance_idx = valuable_starting_points[0]
            else:
                rand_idx = random.randint(0, valuable_starting_points.shape[0] - 1)
                instance_idx = valuable_starting_points[rand_idx]
            bags[bag_idx]['starting_point'][instance_idx] = 1

            # scale is initialized to one
            print('epoch %d, selected instance is from <bag %d, bag label %d, instance %d>. ' %
                  (epoch_idx, bag_idx, bags[bag_idx]['label'], instance_idx), end='')
            [target, scale, nll_cost, init_nll_cost] = self.em(bags,
                                                               scale_indicator,
                                                               bags[bag_idx]['instances'][instance_idx, :],
                                                               np.ones(n_dim, ))
            print('nll before optimization is %f, nll after optimization is %f' % (init_nll_cost, nll_cost))

            targets.append(target)
            scales.append(scale)
            nll_costs.append(nll_cost)

        return targets, scales, nll_costs

    def predict(self, targets, scales, nll_costs, bags, aggregate, threshold):

        n_bag = len(bags)

        bags_label = np.zeros(n_bag, )
        bags_prob = np.zeros(n_bag, )
        instances_prob = list()
        instances_label = list()

        nll_costs = np.asarray(nll_costs)
        targets = np.asarray(targets)
        scales = np.asarray(scales)
        # with maximal negative log likelihood
        if aggregate == 'max':
            target_idx = np.argmax(nll_costs)
            target = targets[target_idx]
            scale = scales[target_idx]
        # with minimal negative log likelihood
        elif aggregate == 'min':
            target_idx = np.argmin(nll_costs)
            target = targets[target_idx]
            scale = scales[target_idx]
        # with average negative log likelihood
        elif aggregate == 'avg':
            target = np.mean(targets, axis=0)
            scale = np.mean(scales, axis=0)
        else:
            raise NotImplementedError('aggregate method must be max, min or avg')

        for bag_idx in range(n_bag):
            instances = bags[bag_idx]['instances']
            dist = np.mean(((instances - target) ** 2) * (scale ** 2), axis=1)
            inst_prob = np.exp(-dist)
            inst_label = np.int8(inst_prob > threshold)
            bags_prob[bag_idx] = np.max(inst_prob)
            bags_label[bag_idx] = np.any(inst_label)
            instances_prob.append(inst_prob)
            instances_label.append(inst_label)

        return bags_label, bags_prob, instances_label, instances_prob
