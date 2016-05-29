
from data_utils import load_kaggle_data_into_instance
from data_utils import parse_mat_data
from feature_utils import extract_fft_frequency_feature
from data_utils import load_kaggle_data_into_bag


def extract_feature_on_kaggle_data(target):
    data_dir = r"I:\ZJU\kaggle\data"
    data_type = 'preictal'
    parse_mat_data(data_dir, target, data_type, extract_fft_frequency_feature)

    data_type = 'interictal'
    parse_mat_data(data_dir, target, data_type, extract_fft_frequency_feature)


if __name__ == '__main__':

    target = 'Patient_1'
    extract_feature_on_kaggle_data(target)

    target = 'Patient_2'

    extract_feature_on_kaggle_data(target)

    target = 'Dog_1'

    extract_feature_on_kaggle_data(target)

    target = 'Dog_2'

    extract_feature_on_kaggle_data(target)

    target = 'Dog_3'

    extract_feature_on_kaggle_data(target)

    target = 'Dog_4'

    extract_feature_on_kaggle_data(target)

    target = 'Dog_5'

    extract_feature_on_kaggle_data(target)

    # load_kaggle_data_into_instance(target)
    # load_kaggle_data_into_bag(target)
