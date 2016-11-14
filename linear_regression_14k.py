import os
import csv
import tensorflow as tf
import numpy as np
import random

data = []

iterations = 5000
batch_size = 100
plot_period = 100
training_range_lower_bound = 12000
training_range_upper_bound = 14000
num_inputs = 7
num_outputs = 1
step_size = 0.000005

muscle_activation = []
f_out = []

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
    x = tf.placeholder(tf.float32, shape=(None, num_inputs))
    W = tf.Variable(tf.zeros([num_inputs,num_outputs]))
    b = tf.Variable(tf.zeros([num_outputs]))

    y = tf.matmul(x,W) + b

    y_ = tf.placeholder(tf.float32, shape=(None,num_outputs))

    cost = tf.reduce_sum(tf.square(y-y_))
    #cost = -tf.reduce_sum(y_*tf.log(y) + (1-y_) * tf.log(1-y))
    #cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, y_))


    W_arr = 0
    bias = 0

    train_step = tf.train.GradientDescentOptimizer(step_size).minimize(cost)
    init = tf.initialize_all_variables()

    sess = tf.Session()
    sess.run(init)

    if os.path.isfile('linear_14k.txt'):
        os.remove('linear_14k.txt')
    f = open('linear_14k.txt', 'w')
    for i in range(iterations): #iterations
        muscle_activation_batch = []
        f_out_batch = []
        for k in range(batch_size):
            temprand = random.randint(1,training_range_lower_bound)
            muscle_activation_batch.append(muscle_activation[temprand])
            f_out_batch.append(f_out[temprand])

        feed = {x: muscle_activation_batch, y_: f_out_batch}
        sess.run(train_step, feed_dict=feed)

        if i%100==0:
            print("After %d iteration:" %i)
            print("W:")
            print(sess.run(W))
            print("b")
            print(sess.run(b))

        W_arr = sess.run(W)
        bias = sess.run(b)[0]

        if i%plot_period==0:
            f.write(str(getError()[0]))
            f.write('\n')




#validation set
def getError():
    print("ERROR: ")

    mean_square_sum_error = 0
    average_percentage_error = 0

    for i in range(training_range_lower_bound, training_range_upper_bound):
        expected_value = float(data[i][18])
        predicted_value = 0
        predicted_muscle_activations = []
        for j in range(15):
            if j%2==0 and j!=0:
                predicted_muscle_activations.append(float(data[i][j]))

        for j,k in enumerate(W_arr):
            predicted_value += predicted_muscle_activations[j]*k

        predicted_value+=bias
        average_percentage_error += abs(predicted_value-expected_value)/expected_value
        mean_square_sum_error += (predicted_value-expected_value)**2
        #print "predicted: " , predicted_value , " actual: " , expected_value

    #mean_square_sum_error/=(training_range_upper_bound-training_range_lower_bound+1)
    average_percentage_error/=(training_range_upper_bound-training_range_lower_bound+1)
    #mean_square_sum_error = mean_square_sum_error/(training_range_upper_bound-training_range_lower_bound+1)

    print("Sum of Square Error: ")
    print(mean_square_sum_error)
    return mean_square_sum_error

    #print "Average Error: "
    #print average_percentage_error

pullData()
trainData()
getError()
