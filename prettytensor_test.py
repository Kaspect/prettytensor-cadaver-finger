import csv
import prettytensor as pt
import tensorflow as tf
import numpy as np
import random
import os
import matplotlib.pyplot as plt

tf.set_random_seed(1234)
np.random.seed(1234)
random.seed(1234)

data = []

iterations = 1500
batch_size = 200
plot_period = 1
training_range_lower_bound = 10000
training_range_upper_bound = 14000
num_inputs = 7
num_outputs = 1
step_size = 0.1

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
    global muscle_activation
    global plot_period

    x = tf.placeholder(tf.float32, shape=(None,num_inputs))
    y = tf.placeholder(tf.float32, shape=(None,num_outputs))

    init_normal = tf.random_normal_initializer(mean=0.0, stddev=0.01)

    loss = (pt.wrap(x)
              .flatten()
              .fully_connected(20, activation_fn=tf.nn.tanh, init=init_normal)
              .fully_connected(20, activation_fn=tf.nn.tanh, init=init_normal)
              #.fully_connected(20, activation_fn=tf.nn.tanh, init=init_normal)
              .fully_connected(num_outputs, activation_fn=tf.nn.tanh, init=init_normal)
              .l2_regression(y))

    optimizer = tf.train.GradientDescentOptimizer(step_size)  # learning rate

    #loss = tf.reduce_sum(tf.square(y-result.apply(x)))

    #loss = pt.l1_regression(result, y)

    train_op = pt.apply_optimizer(optimizer, losses=[loss])


    init_op = tf.initialize_all_variables()

    validation_x = muscle_activation[training_range_lower_bound:]
    validation_y = f_out[training_range_lower_bound:]

    os.remove('pt_14k.txt')
    f = open('pt_14k.txt', 'w')

    i = 0
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



                f.write(str(validation_mse[0]))
                f.write("\n")
                print  '%d: Validation MSE: %g' % (i, validation_mse[0])

'''
            print "ERROR: "

            mean_square_sum_error = 0
            average_percentage_error = 0

            for i in range(training_range_lower_bound, training_range_upper_bound):
                expected_value = float(data[i][18])
                predicted_value = 0
                predicted_muscle_activations = []
                for j in range(15):
                    if j%2==0 and j!=0:
                        predicted_muscle_activations.append(float(data[i][j]))

                predicted_muscle_activations = [predicted_muscle_activations]

                #feed_dict = {x: predicted_muscle_activations}
                #sess.run([loss], y, feed_dict)
'''



'''
    y = tf.matmul(x,W) + b

    y_ = tf.placeholder(tf.float32, shape=(None,num_outputs))

    cost = tf.reduce_sum(tf.square(y-y_))
    #cost = -tf.reduce_sum(y_*tf.log(y) + (1-y_) * tf.log(1-y))
    #cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, y_))



    train_step = tf.train.GradientDescentOptimizer(step_size).minimize(cost)
    init = tf.initialize_all_variables()

    sess = tf.Session()
    sess.run(init)

    for i in range(iterations): #iterations
        muscle_activation_batch = []
        f_out_batch = []
        for k in range(batch_size):
            temprand = random.randint(1,training_range_lower_bound)
            muscle_activation_batch.append(muscle_activation[temprand])
            f_out_batch.append(f_out[temprand])

        feed = {x: muscle_activation_batch, y_: f_out_batch}
        sess.run(train_step, feed_dict=feed)

        print ("After %d iteration:" %i)
        print "W:"
        print sess.run(W)
        print "b"
        print sess.run(b)


    W_arr = sess.run(W)
    bias = sess.run(b)[0]
'''



#validation set
def getError():
    print "ERROR: "

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
        print "predicted: " , predicted_value , " actual: " , expected_value

    mean_square_sum_error/=(training_range_upper_bound-training_range_lower_bound+1)
    average_percentage_error/=(training_range_upper_bound-training_range_lower_bound+1)

    print "Mean Square Sum Error: "
    print mean_square_sum_error

    print "Average Error: "
    print average_percentage_error

pullData()
trainData()
#getError()
f = open('pt_14k.txt', 'r')
data2 = []
for line in f:
    data2.append(line)

plt.plot(data2)
plt.ylabel('MSE')
plt.show()
