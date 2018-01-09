from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import csv
import time
import json
import argparse

import numpy as np
import tensorflow as tf

import mnist_data
from model import MNISTcnn
#from tensorflow.examples.tutorials.mnist import input_data

def predict(sess, x, keep_prob, pred, images, output_file):
    feed_dict = {x:images, keep_prob: 1.0}
    prediction = sess.run(pred, feed_dict=feed_dict)

    with open(output_file, "w") as file:
        writer = csv.writer(file, delimiter = ",")
        writer.writerow(["id","label"])
        for i in range(len(prediction)):
            writer.writerow([str(i), str(prediction[i])])

    print("Output prediction: {0}". format(output_file))


def train(args, data):
    obs_shape = data.train.get_observation_size() # e.g. a tuple (28,28,1)
    assert len(obs_shape) == 3, 'assumed right now'
    #obs_shape = (28,28,1)
    num_class = data.train.labels.shape[1]
    
    x = tf.placeholder(tf.float32, shape=(None,) + obs_shape)
    y = tf.placeholder(tf.float32, (None, num_class))
    model = MNISTcnn(x, y, args)

    optimizer = tf.train.AdamOptimizer(1e-4).minimize(model.loss)

    saver = tf.train.Saver(tf.trainable_variables())

    with tf.Session() as sess:
        print('Starting training')
        sess.run(tf.global_variables_initializer())
        if args.load_params:
            ckpt_file = os.path.join(args.ckpt_dir, 'model.ckpt')
            print('Restoring parameters from', ckpt_file)
            saver.restore(sess, ckpt_file)

        num_batches = data.train.num_examples // args.batch_size
       
        if args.val_size > 0:
            validation = True
            val_num_batches = data.validation.num_examples // args.batch_size
        else:
            validation = False

        for epoch in range(args.epochs):
            begin = time.time()

            # train
            train_accuracies = []
            for i in range(num_batches):
                batch = data.train.next_batch(args.batch_size)
                feed_dict = {x:batch[0], y:batch[1], model.keep_prob: 0.5}
                _, acc = sess.run([optimizer, model.accuracy], feed_dict=feed_dict)
                train_accuracies.append(acc)
            train_acc_mean = np.mean(train_accuracies)


            # compute loss over validation data
            if validation:
                val_accuracies = []
                for i in range(val_num_batches):
                    batch = data.validation.next_batch(args.batch_size)
                    feed_dict = {x:batch[0], y:batch[1], model.keep_prob: 1.0}
                    acc = sess.run(model.accuracy, feed_dict=feed_dict)
                    val_accuracies.append(acc)
                val_acc_mean = np.mean(val_accuracies)

                # log progress to console
                print("Epoch %d, time = %ds, train accuracy = %.4f, validation accuracy = %.4f" % (epoch, time.time()-begin, train_acc_mean, val_acc_mean))
            else:
                print("Epoch %d, time = %ds, train accuracy = %.4f" % (epoch, time.time()-begin, train_acc_mean))
            sys.stdout.flush()

            if (epoch + 1) % 10 == 0:
                ckpt_file = os.path.join(args.ckpt_dir, 'mist_model.ckpt')
                saver.save(sess, ckpt_file)

        ckpt_file = os.path.join(args.ckpt_dir, 'mist_model.ckpt')
        saver.save(sess, ckpt_file)

        # predict test data
        predict(sess, x, model.keep_prob, model.pred, data.test.images, args.output)
        
        
        # origiinal test data from 'http://yann.lecun.com/exdb/mnist/'
        """
        acc = sess.run(model.accuracy, feed_dict={x: data.test.images, y: data.test.labels, model.keep_prob: 1.0})
        print("test accuracy %g"%acc)
        """

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data_dir', type=str, default='mnist_data/', help='Directory for storing input data')
    parser.add_argument('-c', '--ckpt_dir', type=str, default='ckpts/', help='Directory for parameter checkpoints')
    parser.add_argument('-l', '--load_params', dest='load_params', action='store_true', help='Restore training from previous model checkpoint?')
    parser.add_argument("-o", "--output",  type=str, default='prediction.csv', help='Prediction filepath')
    parser.add_argument('-e', '--epochs', type=int, default=30, help='How many epochs to run in total?')
    parser.add_argument('-b', '--batch_size', type=int, default=50, help='Batch size during training per GPU')
    parser.add_argument('-v', '--val_size', type=int, default=5000)
    args = parser.parse_args()
  
    # pretty print args
    print('input args:\n', json.dumps(vars(args), indent=4, separators=(',',':'))) 

    data = mnist_data.read_data_sets(args.data_dir, one_hot=True, reshape=False, validation_size=args.val_size)
    #data = input_data.read_data_sets(args.data_dir, one_hot=True, reshape=False, validation_size=args.val_size)
    
    if not os.path.exists(args.ckpt_dir):
        os.makedirs(args.ckpt_dir)

    train(args, data)