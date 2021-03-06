import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  num_train = X.shape[0]
  num_classes = W.shape[1]
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  for i in range(X.shape[0]):
      scores = X[i].dot(W)
      total_sum = np.sum(np.exp(scores))
      loss += -scores[y[i]] + np.log(total_sum)
      for j in range(num_classes):
          dW[:,j] += (np.exp(W[:,j].dot(X[i])) / float(total_sum)) * X[i]
          if j == y[i]:
            dW[:,j] -= X[i]
  loss /= float(num_train)
  dW /= float(num_train)
  loss += reg * np.sum(W*W)
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################
  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  num_train = X.shape[0]
  num_classes = W.shape[1]
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  scores = X.dot(W) # 500:10
  total_sums = np.sum(np.exp(scores), axis=1) # 500
  loss = np.sum(-scores[np.arange(num_train), y] + np.log(total_sums))
  dW = np.dot(X.T, np.divide(np.exp(scores), total_sums[:, np.newaxis]))

  for i in range(num_train):
    for j in range(num_classes):
      if j == y[i]:
        dW[:,j] -= X[i]
  loss /= float(num_train)
  dW /= float(num_train)
  loss += reg * np.sum(W**2)
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW
