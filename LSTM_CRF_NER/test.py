import tensorflow as tf
import numpy as np

X = np.random.randn(2, 10, 8)
# The second example is of length 6
X[1, 6:] = 0
X_lengths = [10, 6]

cell = tf.nn.rnn_cell.LSTMCell(num_units=5, state_is_tuple=True)

outputs, states = tf.nn.bidirectional_dynamic_rnn(
    cell_fw=cell, cell_bw=cell, dtype=tf.float64, sequence_length=X_lengths, inputs=X
)

output_fw, output_bw = outputs
states_fw, states_bw = states

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    states_shape = tf.shape(states)
    print(states_shape.eval())
    c, h = states_fw
    o = output_fw
    print('c\n', sess.run(c))  #(2,5)
    print('h\n', sess.run(h))  #(2,5)
    print('o\n', sess.run(o))  #(2,10,5)
