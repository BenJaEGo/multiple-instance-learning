
from data_utils import load_kaggle_data_into_instance
from data_utils import parse_mat_data
from feature_utils import get_fft_feature
from feature_utils import transform
from data_utils import load_kaggle_data_into_bag
import os


def extract_feature_on_kaggle_data(target):

    data_dir = r"E:\ZJU\kaggle\data"
    feature_dir = r'\fft_feature'

    current_dir = os.getcwd()

    feature_dir = current_dir + feature_dir

    if os.path.exists(feature_dir):
        print('feature_dir is exist...')
        pass
    else:
        print('feature_dir is not exist...')
        os.mkdir(feature_dir)

    data_type = 'preictal'
    parse_mat_data(data_dir, feature_dir, target, data_type, get_fft_feature)
    # parse_mat_data(data_dir, target, data_type, transform)

    data_type = 'interictal'
    parse_mat_data(data_dir, feature_dir, target, data_type, get_fft_feature)
    # parse_mat_data(data_dir, target, data_type, transform)


if __name__ == '__main__':

    target = 'Dog_1'
    extract_feature_on_kaggle_data(target)

    # target = 'Patient_2'
    #
    # extract_feature_on_kaggle_data(target)
    #
    # target = 'Dog_1'
    #
    # extract_feature_on_kaggle_data(target)
    #
    # target = 'Dog_2'
    #
    # extract_feature_on_kaggle_data(target)
    #
    # target = 'Dog_3'
    #
    # extract_feature_on_kaggle_data(target)
    #
    # target = 'Dog_4'
    #
    # extract_feature_on_kaggle_data(target)
    #
    # target = 'Dog_5'
    #
    # extract_feature_on_kaggle_data(target)

    # load_kaggle_data_into_instance(target)
    # load_kaggle_data_into_bag(target)
