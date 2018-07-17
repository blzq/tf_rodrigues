#!/usr/bin/python3
# -*- encoding: utf-8 -*-

import tensorflow as tf


def rodrigues_batch(rvecs):
    """
    Convert a batch of axis-angle rotations in rotation vector form shaped
    (batch, 3) to a batch of rotation matrices shaped (batch, 3, 3).
    See
    https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula#Matrix_notation
    https://en.wikipedia.org/wiki/Rotation_matrix#Rotation_matrix_from_axis_and_angle
    """
    batch_size = tf.shape(rvecs)[0]
    tf.assert_equal(tf.shape(rvecs)[1], 3)

    thetas = tf.norm(rvecs, axis=1, keepdims=True)
    is_zero = tf.equal(tf.squeeze(thetas), 0.0)
    u = rvecs / thetas

    # Each K is the cross product matrix of unit axis vectors
    # pyformat: disable
    zero = tf.zeros([batch_size])  # for broadcasting
    Ks_1 = tf.stack([  zero   , -u[:, 2],  u[:, 1] ], axis=1)  # row 1
    Ks_2 = tf.stack([  u[:, 2],  zero   , -u[:, 0] ], axis=1)  # row 2
    Ks_3 = tf.stack([ -u[:, 1],  u[:, 0],  zero    ], axis=1)  # row 3
    # pyformat: enable
    Ks = tf.stack([Ks_1, Ks_2, Ks_3], axis=2)                  # stack rows

    Rs = tf.eye(3, batch_shape=[batch_size]) + \
         tf.sin(thetas)[..., tf.newaxis] * Ks + \
         (1 - tf.cos(thetas)[..., tf.newaxis]) * tf.matmul(Ks, Ks)

    # Avoid returning NaNs where division by zero happened
    return tf.where(is_zero,
                    tf.eye(3, batch_shape=[batch_size]), Rs)


# For testing only
if __name__ == '__main__':
    import numpy as np
    import cv2
    rvecs = np.random.randn(4, 12).astype(np.float32)

    rvecs_tf = tf.constant(rvecs)
    Rs = rodrigues_batch(tf.reshape(rvecs_tf, [-1, 3]))
    with tf.Session() as sess:
        print("TensorFlow: ")
        print(sess.run(Rs))

    print("\nOpenCV: ")
    for rvec in np.reshape(rvecs, [-1, 3]):
        mat, _ = cv2.Rodrigues(rvec)
        print(mat.T)
        print()
