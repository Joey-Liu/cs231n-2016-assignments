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

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_train = X.shape[0]
  num_classes = W.shape[1]
  scores = X.dot(W)
  scores_max_perrow = np.max(scores, axis=1)
  scores_max_perrow = np.reshape(scores_max_perrow, [num_train, 1])
  # print scores_max_perrow.shape
  # print scores.shape
  scores -= scores_max_perrow
  scores = np.exp(scores)
  for i in range(num_train):
    loss -= np.log(scores[i, y[i]] / np.sum(scores[i]))
    for j in range(num_classes):
      if j == y[i]:
        dW[:, j] += -X[i] + scores[i, y[i]] / np.sum(scores[i]) * X[i]
      else:
        dW[:, j] += scores[i, j] / np.sum(scores[i]) * X[i]
  loss /= num_train
  loss += np.sum(W * W) * reg * 0.5

  dW /= num_train
  dW += reg * W
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

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  # LOSS
  scores = X.dot(W)
  scores_max_perrow = np.max(scores, axis=1)
  scores_max_perrow = np.reshape(scores_max_perrow, [num_train, 1])
  scores -= scores_max_perrow
  scores = np.exp(scores)
  probabilities = -np.log(scores[np.arange(num_train), y] / np.sum(scores,axis=1))
  loss += np.sum(probabilities)
  loss /= num_train
  loss += 0.5 * reg * np.sum(W * W)

  # dW
  normal_scores = scores / np.reshape(np.sum(scores, axis=1), [num_train, 1])
  normal_scores[np.arange(num_train), y] -= 1
  dW = X.transpose().dot(normal_scores)
  dW /= num_train
  dW += reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

