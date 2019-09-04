import tensorflow as tf
import numpy as np

input_x = np.array([
    [5, 2, -1],
    [2, 5, 3],
    [-1, 3, 5],
    [3, 4, 5]
])


def calc_2_moments(tensor):
    """flattens tensor and calculates sample mean and covariance matrix 
    along last dim (presumably channels)"""

    shape = tf.shape(tensor, out_type=tf.int32)
    n = tf.reduce_prod(shape[:-1])

    flat_array = tf.reshape(tensor, (n, shape[-1]))
    mu = tf.reduce_mean(flat_array, axis=0, keepdims=True)
    cov = (tf.matmul(flat_array - mu, flat_array - mu, transpose_a=True) /
           tf.cast(n, tf.float32))

    return mu, cov


input_tensor = tf.placeholder(
    shape=(4, 3), dtype=tf.float32
)
mu, cov = calc_2_moments(input_tensor)
eigvals, eigvects = tf.self_adjoint_eig(cov)
eigroot_mat = tf.diag(tf.sqrt(tf.maximum(eigvals, 0.)))
root_cov = tf.matmul(tf.matmul(eigvects, eigroot_mat),
                     eigvects, transpose_b=True)
tr_cov = tf.reduce_sum(tf.maximum(eigvals, 0))

with tf.Session() as sess:
    mu_, root_cov_, tr_cov_, eigvals_, eigroot_mat_ = sess.run(
        [mu, root_cov, tr_cov, eigvals, eigroot_mat], feed_dict={
            input_tensor: input_x}
    )
    print(mu_)
    print(eigvals_)
    print(eigroot_mat_)
    print(tr_cov_)
    print(root_cov_)
