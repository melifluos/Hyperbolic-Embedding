"""
Tests for negative sampling. Trying to understand why model performance decays rapidly with batch size
"""
import numpy as np

num_samples = 5
data = np.array([0, 0, 0, 1, 1, 0, 2, 2, 1, 2, 3, 3, 3, 1])
uniques, vocab_counts = np.unique(data, return_counts=True)
vocab_size = len(uniques)
labels = np.array([1, 2])


def unigram_sample(labels_matrix, num_samples, vocab_size, vocab_counts):
    sampled_ids, _, _ = (tf.nn.fixed_unigram_candidate_sampler(
        true_classes=labels_matrix,
        num_true=1,
        num_sampled=num_samples,
        unique=True,
        range_max=vocab_size,
        distortion=0.75,
        unigrams=vocab_counts.tolist()))

    return sampled_ids
