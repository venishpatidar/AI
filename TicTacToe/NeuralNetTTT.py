import tensorflow as tf
import numpy as np
from tensorflow.python.framework import ops

def create_placeholders(n_x,n_y):
    X = tf.placeholder(shape=[n_x,None], size=tf.float32,name="X")
    Y = tf.placeholder(shape=[n_y,None],size=tf.float32,name="Y")
    return X,Y

def initiliaze_parameters():
    W1 = tf.get_variable("W1",[6,9],initializer = tf.contrib.layers.xavier_initializer())
    b1 = tf.get_variable("b1",[6,1],initializer = tf.zeros_initializer())
    W2 = tf.get_variable("W2",[3,6],initializer = tf.contrib.layers.xavier_initializer())
    b2 = tf.get_variable("b2",[3,1],initializer = tf.zeros_initializer())
    W3 = tf.get_variable("W3",[2,2],initializer = tf.contrib.layers.xavier_initializer())
    b3 = tf.get_variable("b3",[2,1],initializer=tf.zeros_initializer())

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2,
                  "W3": W3,
                  "b3": b3}
    return parameters

def forward_propagation(X, parameters):
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    W3 = parameters['W3']
    b3 = parameters['b3']
    
    Z1 = tf.add(tf.matmul(W1,X),b1)                                              
    A1 = tf.nn.relu(Z1)                                              
    Z2 = tf.add(tf.matmul(W2,A1),b2)                                              
    A2 = tf.nn.relu(Z2)                                              
    Z3 = tf.add(tf.matmul(W3,A2),b3)

    return Z3

def compute_cost(Z3, Y):
    logits = tf.transpose(Z3)
    labels = tf.transpose(Y)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = logits , labels = labels))
    return cost
def make_minibatches():
    pass

def model(X_train,Y_train,X_test,Y_test,learning_rate = 0.001,num_epochs=100,minibatch_size=3,print_cost=True):
    (n_x,m) = X_train.shape
    n_y = Y_train.shape[0]
    costs = []

    X,Y = create_placeholders(n_x,n_y)
    parameters = initiliaze_parameters()
    Z3 = forward_propagation(X,parameters)

    cost = compute_cost(Z3, Y)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)      
    
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        for epoch in num_epochs:

            _,cost= sess.run([optimizer,cost],feed_dict={X:X_train})

X = np.load("X",dtype=int)
print(X)