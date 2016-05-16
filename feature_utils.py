import numpy as np


def extract_fft_frequency_feature(segment):

    n_slice_time = 5
    low_frequency = 16
    high_frequency = 32

    data = segment.data
    sampling_frequency = segment.sampling_frequency
    data_length_sec = segment.data_length_sec
    slice_length = int(np.floor(n_slice_time * sampling_frequency))
    n_slice = int(data_length_sec / n_slice_time)

    feature = list()

    for slice_idx in range(n_slice):
        slice_data = data[:, slice_idx * slice_length:(slice_idx + 1) * slice_length]
        fft_slice_data = abs(np.fft.fft(slice_data, axis=1))
        slice_feature = fft_slice_data[:, low_frequency:high_frequency]
        slice_feature = slice_feature.reshape(1, np.prod(slice_feature.shape))
        feature.append(slice_feature)
    return feature


