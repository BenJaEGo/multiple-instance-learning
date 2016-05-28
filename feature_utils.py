import numpy as np


def extract_fft_frequency_feature(segment, n_slice_time=1, low_frequency=1,  high_frequency=48):
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

    # print(sampling_frequency)
    slice_length = int(np.floor(n_slice_time * sampling_frequency))
    # print(slice_length)
    n_slice = int(data_length_sec / n_slice_time)

    feature = list()

    for slice_idx in range(n_slice):
        slice_data = data[:, slice_idx * slice_length:(slice_idx + 1) * slice_length]
        fft_slice_data = np.absolute(np.fft.rfft(slice_data, axis=1))
        slice_feature = fft_slice_data[:, low_frequency:high_frequency]
        slice_feature = np.log10(slice_feature)
        slice_feature = slice_feature.reshape(np.prod(slice_feature.shape), )
        feature.append(slice_feature)
    return feature


