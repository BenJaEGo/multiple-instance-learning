import numpy as np
from sklearn import preprocessing
from scipy import signal


def upper_right_triangle(matrix):
    accum = []
    for i in range(matrix.shape[0]):
        for j in range(i + 1, matrix.shape[1]):
            accum.append(matrix[i, j])
    return np.array(accum)


def fft(data, low_frequency, high_frequency):
    fft_slice_data = np.absolute(np.fft.rfft(data, axis=1))
    slice_fft_feature = fft_slice_data[:, low_frequency:high_frequency]
    slice_fft_feature = np.log10(slice_fft_feature)
    return slice_fft_feature


def freq_corr(fft_data):
    scaled_data = preprocessing.scale(fft_data, axis=0)
    correlation_matrix = np.corrcoef(scaled_data)
    eigenvalues = np.absolute(np.linalg.eig(correlation_matrix)[0])
    eigenvalues.sort()
    correlation_coef = upper_right_triangle(correlation_matrix)
    correlation_feature = np.concatenate((correlation_coef, eigenvalues))
    return correlation_feature


def time_corr(data):
    resampled_data = signal.resample(data, 400, axis=1) if data.shape[-1] > 400 else data
    scaled_data = preprocessing.scale(resampled_data, axis=0)
    correlation_matrix = np.corrcoef(scaled_data)
    eigenvalues = np.absolute(np.linalg.eig(correlation_matrix)[0])
    eigenvalues.sort()
    correlation_coef = upper_right_triangle(correlation_matrix)
    correlation_feature = np.concatenate((correlation_coef, eigenvalues))
    return correlation_feature


def get_fft_feature(segment, n_slice_time=1, low_frequency=1,  high_frequency=48):
    """
        Fast Fourier Transform is applied to each n_slice_time seconds clip across all EEG channels,
        taking log10 of the magnitudes of frequencies in the range (low_frequency, high_frequency).
        :param segment: raw_data of kaggle
        :param n_slice_time: each clip length (second) of signals
        :param low_frequency: low frequency, usually ignore 0 and begin with 1
        :param high_frequency: high frequency, up to high_frequency-1
        :return: feature after pre-process
    """
    data = segment.data
    sampling_frequency = segment.sampling_frequency
    data_length_sec = segment.data_length_sec

    slice_length = int(np.floor(n_slice_time * sampling_frequency))
    n_slice = int(data_length_sec / n_slice_time)

    feature = list()

    for slice_idx in range(n_slice):
        slice_data = data[:, slice_idx * slice_length:(slice_idx + 1) * slice_length]
        slice_fft_feature = fft(slice_data, low_frequency, high_frequency)
        feature.append(slice_fft_feature.ravel())

    return feature


def transform(segment, n_slice_time=1, low_frequency=1,  high_frequency=48):
    data = segment.data
    sampling_frequency = segment.sampling_frequency
    data_length_sec = segment.data_length_sec

    slice_length = int(np.floor(n_slice_time * sampling_frequency))
    n_slice = int(data_length_sec / n_slice_time)

    feature = list()

    for slice_idx in range(n_slice):
        slice_data = data[:, slice_idx * slice_length:(slice_idx + 1) * slice_length]
        slice_fft_feature = fft(slice_data, low_frequency, high_frequency)
        freq_corr_feature = freq_corr(slice_fft_feature)
        time_corr_feature = time_corr(slice_data)

        slice_feature = np.concatenate((slice_fft_feature.ravel(), freq_corr_feature, time_corr_feature))

        feature.append(slice_feature)
    return feature
