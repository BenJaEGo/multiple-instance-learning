
from data_utils import load_kaggle_data_into_instance
from data_utils import parse_mat_data
from feature_utils import extract_fft_frequency_feature
from data_utils import load_kaggle_data_into_bag


def extract_feature_on_kaggle_data():
    data_dir = r"E:\ZJU\kaggle\data"
    target = 'Patient_1'
    data_type = 'preictal'
    parse_mat_data(data_dir, target, data_type, extract_fft_frequency_feature)

    data_type = 'interictal'
    parse_mat_data(data_dir, target, data_type, extract_fft_frequency_feature)


if __name__ == '__main__':
    # extract_feature_on_kaggle_data()


    # target = 'Dog_1'
    target = 'Patient_1'
    # feature, label = load_kaggle_data_into_instance(target)
    load_kaggle_data_into_bag(target)
