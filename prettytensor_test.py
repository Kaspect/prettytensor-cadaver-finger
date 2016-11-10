import csv
import prettytensor as pt
import tensorflow as tf
import numpy as np
import random
import os
import matplotlib.pyplot as plt
import ipdb

import multiprocessing as multi
from functools import partial
#from multiprocessing import Manager


tf.set_random_seed(1234)
np.random.seed(1234)
random.seed(1234)

data = []

iterations = 100
batch_size = 200
plot_period = 1
training_range_lower_bound = 10000
training_range_upper_bound = 14000
num_inputs = 7
num_outputs = 1
step_size = 0.1

muscle_activation = []
f_out = []

def print_update_message_to_console(value_label, value, iteration_label, iteration):
    print('%s=%d\t%s=%f' % (iteration_label, iteration, value_label, value))

def append_float_to_existing_file(file_writer, float_value):
    file_writer.write(str(float_value))
    file_writer.write('\n')

def plot_mse_convergence(file_name):
    f = open(file_name, 'r')
    mse_for_each_successive_iteration = []
    for line in f:
        mse_for_each_successive_iteration.append(line)

    plt.plot(mse_for_each_successive_iteration)
    plt.ylabel('Mean Squared Error (f_prediction - f_experimental)^2')
    plt.xlabel('Iteration')
    plt.show()

def histogram_of_force_absolute_diff(list_of_unidimensional_floats, breaks):
    #bins = np.arange(-100, 100, 5) # fixed bin size
    bins = np.linspace(min(list_of_unidimensional_floats), max(list_of_unidimensional_floats), breaks)

    plt.hist(list_of_unidimensional_floats, bins=bins, alpha=0.5)
    plt.title('Distribution of Squared Errors of Fx (fixed bin size)')
    plt.xlabel(str('variable |f_prediction - f_experimental|^2 (bin size = %f)' % ((0.005-0)/breaks)))
    plt.ylabel('Validation Set Observations (count)')

    plt.show()

def compute_list_of_mse_values_per_iteration_per_datapoint_helper(sess):
    def compute_list_of_mse_values_per_iteration_per_datapoint(input):
        return (sess.run([loss], {x: [input[0]], y: [input[1]]}))
    return compute_list_of_mse_values_per_iteration_per_datapoint

def pullData():
    global data
    with open('data/matrix.csv') as csvfile:
        rowReader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        for row in rowReader:
            data.append(', '.join(row).split(','))

    data = data[1:]

    #muscle_activation = np.array([])
    #f_out = np.array([])

    for i in data:
        #a = np.array([i[2], i[4], i[6], i[8], i[10], i[12], i[14]])
        #np.append(muscle_activation, a)
        #np.append(f_out, i[18])
        a = []
        for j in range(15):
            if j%2==0 and j!=0:
                a.append(i[j])
        muscle_activation.append(a)
        #muscle_activation.append([tf.placeholder(i[2]), i[4], i[6], i[8], i[10], i[12], i[14]])
        #f_out.append([i[18], i[16], i[21], i[17], i[20], i[19]]) # F_x, F_y, F_z, M_x, M_y, M_z
        f_out.append([i[18]])

def trainData():
    global W_arr
    global bias
    global muscle_activation
    global plot_period

    x = tf.placeholder(tf.float32, shape=(None,num_inputs))
    y = tf.placeholder(tf.float32, shape=(None,num_outputs))

    init_normal = tf.random_normal_initializer(mean=0.0, stddev=0.01)

    loss = (pt.wrap(x)
              .flatten()
              .fully_connected(20, activation_fn=tf.nn.tanh, init=init_normal, name='layer1')
              .fully_connected(20,  activation_fn=tf.nn.tanh, init=init_normal, name='layer2')
              #.fully_connected(20, activation_fn=tf.nn.tanh, init=init_normal)
              .fully_connected(num_outputs, activation_fn=tf.nn.tanh, init=init_normal, name='output_layer')
              .l2_regression(y))

    optimizer = tf.train.GradientDescentOptimizer(step_size)  # learning rate

    #loss = tf.reduce_sum(tf.square(y-result.apply(x)))

    #loss = pt.l1_regression(result, y)

    train_op = pt.apply_optimizer(optimizer, losses=[loss])


    init_op = tf.initialize_all_variables()

    validation_x = muscle_activation[training_range_lower_bound:]
    validation_y = f_out[training_range_lower_bound:]
    validations = zip(validation_x, validation_y)

    if os.path.isfile('pt_14k.txt'):
        os.remove('pt_14k.txt')
    f = open('pt_14k.txt', 'w')

    i = 0

    error_heat_map = []

    with tf.Session() as sess:
        sess.run(init_op)
        for i in range(iterations):
            #generate random batches
            muscle_activation_batch = []
            f_out_batch = []
            for k in range(batch_size):
                temprand = random.randint(1,training_range_lower_bound)
                muscle_activation_batch.append(muscle_activation[temprand])
                f_out_batch.append(f_out[temprand])


            mse = sess.run([train_op, loss],
                                     {x: muscle_activation_batch, y: f_out_batch})

            #print 'Loss: %g' % mse[1]

            if i%plot_period==0:
                validation_mse = sess.run([loss],
                                        {x: validation_x, y: validation_y})
                append_float_to_existing_file(f, validation_mse[0])
                print_update_message_to_console('MSE', validation_mse[0], 'Iteration', i)

            # intput: i, validation_x, validation_y
            # output: sess.run([loss], {x: [validation_x[input]], y: [validation_y[input]]})
            f_x_output = map(lambda input : sess.run([loss], {x: [input[0]], y: [input[1]]}) , validations)
            error_heat_map.append(f_x_output)


            '''
            # Used to return weights for each layer
            with tf.variable_scope('layer1', reuse=True):
                print sess.run(tf.get_variable('weights'))
            with tf.variable_scope('layer2', reuse=True):
                print sess.run(tf.get_variable('weights'))
            with tf.variable_scope('output_layer', reuse=True):
                print sess.run(tf.get_variable('weights'))

            '''
    histogram_of_force_absolute_diff([x[0] for x in error_heat_map[0]], 10)
    f.close()




pullData()
trainData()

plot_mse_convergence(file_name='pt_14k.txt')
'''

'''
