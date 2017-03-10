from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector

from process_data import process_data

VOCAB_SIZE = 50000
BATCH_SIZE = 128
EMBED_SIZE = 128 # dimension of the word embedding vectors
SKIP_WINDOW = 1 # the context window
NUM_SAMPLED = 64    # Number of negative examples to sample.
LEARNING_RATE = 1.0
NUM_TRAIN_STEPS = 10000
SKIP_STEP = 2000 # how many steps to skip before reporting the loss


class Word2VecModel:

    def __init__(self, vocab_size, batch_size, embed_size, num_sampled, learning_rate):

        self.batch_size = batch_size
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.num_sampled = num_sampled
        self.learning_rate = learning_rate

        self._build_graph()

    def _create_placeholders(self):
        with tf.name_scope('data'):
            self.center_words = tf.placeholder(tf.int32, shape=[self.batch_size], name='center_words')
            self.target_words = tf.placeholder(tf.int32, shape=[self.batch_size, 1], name='target_words')

    def _create_embedding(self):
        with tf.name_scope('embed'):
            self.embed_matrix = tf.Variable(tf.random_uniform([
                self.vocab_size, self.embed_size], -1.0, 1.0), name='embed_matrix')

    def _create_loss(self):
        with tf.name_scope('loss'):
            self.embed = tf.nn.embedding_lookup(self.embed_matrix, self.center_words, name='embed')
            self.nce_weight = tf.Variable(tf.truncated_normal([self.vocab_size, self.embed_size],
                            stddev=1.0 / self.embed_size ** 0.5), name='nce_weight')
            self.nce_bias = tf.Variable(tf.zeros([self.vocab_size]), name='nce_bias')
            self.loss = tf.reduce_mean(tf.nn.nce_loss(
                                                weights=self.nce_weight,
                                                biases=self.nce_bias,
                                                labels=self.target_words,
                                                inputs=self.embed,
                                                num_sampled=self.num_sampled,
                                                num_classes=self.vocab_size),
                                                name='loss')

    def _create_optimizer(self):
        self.optimizer = tf.train.GradientDescentOptimizer(
            self.learning_rate).minimize(self.loss, global_step=self.global_step)

    def _build_graph(self):
        self.global_step = tf.Variable(0, dtype=tf.int32, trainable=False,
            name='global_step')
        self._create_placeholders()
        self._create_embedding()
        self._create_loss()
        self._create_optimizer()



def word2vec(batch_gen):
    """ Build the graph for word2vec model and train it """

    with tf.device('/cpu:0'):
        model = Word2VecModel(vocab_size=VOCAB_SIZE,
            batch_size=BATCH_SIZE,
            embed_size=EMBED_SIZE,
            num_sampled=NUM_SAMPLED,
            learning_rate=LEARNING_RATE)

    saver = tf.train.Saver()

    RESTORE_SESSION = True
    SAVE_SESSION = True

    with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
        sess.run(tf.global_variables_initializer())

        if RESTORE_SESSION:
            ckpt = tf.train.get_checkpoint_state(os.path.dirname('./checkpoints/checkpoint'))
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)

        total_loss = 0.0 # we use this to calculate the average loss in the last SKIP_STEP steps
        writer = tf.summary.FileWriter('./my_graph/no_frills/', sess.graph)
        for index in range(NUM_TRAIN_STEPS):
            centers, targets = next(batch_gen)
            loss_batch, __ = sess.run([model.loss, model.optimizer],
                                feed_dict={model.center_words: centers,
                                            model.target_words: targets})

            total_loss += loss_batch
            if (index + 1) % SKIP_STEP == 0:
                if SAVE_SESSION:
                    saver.save(sess, './checkpoints/word2vec', global_step=model.global_step)
                print('Average loss at step {}: {:5.1f}'.format(index, total_loss / SKIP_STEP))
                total_loss = 0.0

        ### Embeding writer
        # final_embed_matrix = sess.run(model.embed_matrix)
        # embedding_var = tf.Variable(final_embed_matrix[:500], name='embedding')
        # sess.run(embedding_var.initializer)
        # config = projector.ProjectorConfig()
        # summary_writer = tf.summary.FileWriter('./my_graph/no_frills/')
        # embedding = config.embeddings.add()
        # embedding.tensor_name = embedding_var.name
        #
        # embedding.metadata_path + './my_graph/no_frills/' + 'vocab_500.tsv'
        # print('===================\n\n\n HERE \n\n\n===============')
        # projector.visualize_embeddings(summary_writer, config)
        #
        # saver_embed = tf.train.Saver([embedding_var])
        # saver_embed.save(sess, './my_graph/no_frills/' + 'skip-gram.cpkt', 1)
        ###

        writer.close()


def main():
    batch_gen = process_data(VOCAB_SIZE, BATCH_SIZE, SKIP_WINDOW)
    word2vec(batch_gen)


if __name__ == '__main__':
    main()
