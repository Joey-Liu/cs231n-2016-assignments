import numpy as np


def affine_forward(x, w, b):
  """
  Computes the forward pass for an affine (fully-connected) layer.

  The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
  examples, where each example x[i] has shape (d_1, ..., d_k). We will
  reshape each input into a vector of dimension D = d_1 * ... * d_k, and
  then transform it to an output vector of dimension M.

  Inputs:
  - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
  - w: A numpy array of weights, of shape (D, M)
  - b: A numpy array of biases, of shape (M,)
  
  Returns a tuple of:
  - out: output, of shape (N, M)
  - cache: (x, w, b)
  """
  out = None
  #############################################################################
  # TODO: Implement the affine forward pass. Store the result in out. You     #
  # will need to reshape the input into rows.                                 #
  #############################################################################
  x_t = np.reshape(x, [x.shape[0], -1])
  out = np.dot(x_t, w) + b
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  cache = (x, w, b)
  return out, cache


def affine_backward(dout, cache):
  """
  Computes the backward pass for an affine layer.

  Inputs:
  - dout: Upstream derivative, of shape (N, M)
  - cache: Tuple of:
    - x: Input data, of shape (N, d_1, ... d_k)
    - w: Weights, of shape (D, M)

  Returns a tuple of:
  - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
  - dw: Gradient with respect to w, of shape (D, M)
  - db: Gradient with respect to b, of shape (M,)
  """
  x, w, b = cache
  dx, dw, db = None, None, None
  #############################################################################
  # TODO: Implement the affine backward pass.                                 #
  #############################################################################
  dx = dout.dot(w.transpose())
  dx = np.reshape(dx, x.shape)

  x_t = np.reshape(x, [x.shape[0], -1])
  dw = x_t.transpose().dot(dout)

  db = dout.transpose().dot(np.ones([dx.shape[0], 1]))
  db = np.squeeze(db)
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dx, dw, db


def relu_forward(x):
  """
  Computes the forward pass for a layer of rectified linear units (ReLUs).

  Input:
  - x: Inputs, of any shape

  Returns a tuple of:
  - out: Output, of the same shape as x
  - cache: x
  """
  out = None
  #############################################################################
  # TODO: Implement the ReLU forward pass.                                    #
  #############################################################################
  out = np.maximum(x, 0)
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  cache = x
  return out, cache


def relu_backward(dout, cache):
  """
  Computes the backward pass for a layer of rectified linear units (ReLUs).

  Input:
  - dout: Upstream derivatives, of any shape
  - cache: Input x, of same shape as dout

  Returns:
  - dx: Gradient with respect to x
  """
  dx, x = None, cache
  #############################################################################
  # TODO: Implement the ReLU backward pass.                                   #
  #############################################################################
  index = np.nonzero(np.maximum(x, 0))
  t = np.zeros(x.shape)
  t[index] = 1
  dx = t * dout
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dx


def batchnorm_forward(x, gamma, beta, bn_param):
  """
  Forward pass for batch normalization.
  
  During training the sample mean and (uncorrected) sample variance are
  computed from minibatch statistics and used to normalize the incoming data.
  During training we also keep an exponentially decaying running mean of the mean
  and variance of each feature, and these averages are used to normalize data
  at test-time.

  At each timestep we update the running averages for mean and variance using
  an exponential decay based on the momentum parameter:

  running_mean = momentum * running_mean + (1 - momentum) * sample_mean
  running_var = momentum * running_var + (1 - momentum) * sample_var

  Note that the batch normalization paper suggests a different test-time
  behavior: they compute sample mean and variance for each feature using a
  large number of training images rather than using a running average. For
  this implementation we have chosen to use running averages instead since
  they do not require an additional estimation step; the torch7 implementation
  of batch normalization also uses running averages.

  Input:
  - x: Data of shape (N, D)
  - gamma: Scale parameter of shape (D,)
  - beta: Shift paremeter of shape (D,)
  - bn_param: Dictionary with the following keys:
    - mode: 'train' or 'test'; required
    - eps: Constant for numeric stability
    - momentum: Constant for running mean / variance.
    - running_mean: Array of shape (D,) giving running mean of features
    - running_var Array of shape (D,) giving running variance of features

  Returns a tuple of:
  - out: of shape (N, D)
  - cache: A tuple of values needed in the backward pass
  """
  mode = bn_param['mode']
  eps = bn_param.get('eps', 1e-5)
  momentum = bn_param.get('momentum', 0.9)

  N, D = x.shape
  running_mean = bn_param.get('running_mean', np.zeros(D, dtype=x.dtype))
  running_var = bn_param.get('running_var', np.zeros(D, dtype=x.dtype))

  out, cache = None, None
  if mode == 'train':
    #############################################################################
    # TODO: Implement the training-time forward pass for batch normalization.   #
    # Use minibatch statistics to compute the mean and variance, use these      #
    # statistics to normalize the incoming data, and scale and shift the        #
    # normalized data using gamma and beta.                                     #
    #                                                                           #
    # You should store the output in the variable out. Any intermediates that   #
    # you need for the backward pass should be stored in the cache variable.    #
    #                                                                           #
    # You should also use your computed sample mean and variance together with  #
    # the momentum variable to update the running mean and running variance,    #
    # storing your result in the running_mean and running_var variables.        #
    #############################################################################
    x_mean = x.mean(axis=0)
    x_var = x.var(axis=0)
    h1 = (x - x_mean)
    h2 = np.sqrt(x_var + eps)
    normalized_x = h1 / h2

    out = gamma * normalized_x + beta
    running_mean = momentum * running_mean + (1 - momentum) * x_mean
    running_var = momentum * running_var + (1 - momentum) * x_var

    cache = {}
    cache['x'] = x
    cache['x_mean'] = x_mean
    cache['x_var'] = x_var
    cache['h1'] = h1
    cache['h2'] = h2
    cache['eps'] = eps
    cache['normalized_x'] = normalized_x
    cache['gamma'] = gamma
    cache['beta'] = beta
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
  elif mode == 'test':
    #############################################################################
    # TODO: Implement the test-time forward pass for batch normalization. Use   #
    # the running mean and variance to normalize the incoming data, then scale  #
    # and shift the normalized data using gamma and beta. Store the result in   #
    # the out variable.                                                         #
    #############################################################################
    h1 = x - running_mean
    h2 = np.sqrt(running_var) + eps
    out = h1 / h2
    out = out * gamma + beta

    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
  else:
    raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

  # Store the updated running means back into bn_param
  bn_param['running_mean'] = running_mean
  bn_param['running_var'] = running_var

  return out, cache


def batchnorm_backward(dout, cache):
  """
  Backward pass for batch normalization.
  
  For this implementation, you should write out a computation graph for
  batch normalization on paper and propagate gradients backward through
  intermediate nodes.
  
  Inputs:
  - dout: Upstream derivatives, of shape (N, D)
  - cache: Variable of intermediates from batchnorm_forward.
  
  Returns a tuple of:
  - dx: Gradient with respect to inputs x, of shape (N, D)
  - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
  - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
  """
  dx, dgamma, dbeta = None, None, None
  #############################################################################
  # TODO: Implement the backward pass for batch normalization. Store the      #
  # results in the dx, dgamma, and dbeta variables.                           #
  #############################################################################
  N = dout.shape[0]
  h1 = cache['h1']
  h2 = cache['h2']
  gamma = cache['gamma']
  x_var = cache['x_var']
  normalized_x = cache['normalized_x']
  eps = cache['eps']
  x = cache['x']
  x_mean = cache['x_mean']

  dgamma = (dout * normalized_x).sum(axis=0)
  dbeta = dout.sum(axis=0)

  dnormalized_x = gamma * dout # N * D
  # h1 and h2 layer, you might want to read the graph to see what's going on
  dh1 = 1 / h2 * dnormalized_x
  dh2 = (-h1 / np.square(h2) * dnormalized_x).sum(axis=0)

  # mean layer and variance layer
  dmean = (-1 * dh1).sum(axis=0)
  dvar = 0.5 / np.sqrt(x_var + eps) * dh2

  # x layer, you might want to read to the graph to see what's going on
  dx = (dmean / N + dh1 * 1) + 2./N * (x - x_mean) * dvar

  # dx_dh1 = 1.0 / h2
  # dx_dh2 = -h1 / np.square(h2)
  #
  # dh1_divide_dmean = -1
  # dmean_divide_dx = 1.0 / N
  # dfh1_divide_dx = 1
  # dh1_divide_dx = dh1_divide_dmean * dmean_divide_dx + dfh1_divide_dx
  #
  # dh2_divide_dvar = 0.5 / np.sqrt(x_var + eps)
  # dvar_divide_dx = 2.0 / N * (x - x_mean)
  # dh2_divide_dx = dh2_divide_dvar * dvar_divide_dx
  #
  # dx = (dx_dh1 * dh1_divide_dx + dx_dh2 * dh2_divide_dx) * dnormalized_x
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return dx, dgamma, dbeta


def batchnorm_backward_alt(dout, cache):
  """
  Alternative backward pass for batch normalization.
  
  For this implementation you should work out the derivatives for the batch
  normalizaton backward pass on paper and simplify as much as possible. You
  should be able to derive a simple expression for the backward pass.
  
  Note: This implementation should expect to receive the same cache variable
  as batchnorm_backward, but might not use all of the values in the cache.
  
  Inputs / outputs: Same as batchnorm_backward
  """
  dx, dgamma, dbeta = None, None, None
  #############################################################################
  # TODO: Implement the backward pass for batch normalization. Store the      #
  # results in the dx, dgamma, and dbeta variables.                           #
  #                                                                           #
  # After computing the gradient with respect to the centered inputs, you     #
  # should be able to compute gradients with respect to the inputs in a       #
  # single statement; our implementation fits on a single 80-character line.  #
  #############################################################################
  # dgamma = (cache['normalized_x'] * dout).sum(axis=0)
  # dbeta = dout.sum(axis=0)
  # S = cache['x_var'] + cache['eps']
  # n = dout.shape[0]
  # dx = cache['gamma'] * dout * (1. - 1 / n - np.square(cache['x'] - cache['x_mean']) / (n * S)) / np.sqrt(S)
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  
  return dx, dgamma, dbeta


def dropout_forward(x, dropout_param):
  """
  Performs the forward pass for (inverted) dropout.

  Inputs:
  - x: Input data, of any shape
  - dropout_param: A dictionary with the following keys:
    - p: Dropout parameter. We drop each neuron output with probability p.
    - mode: 'test' or 'train'. If the mode is train, then perform dropout;
      if the mode is test, then just return the input.
    - seed: Seed for the random number generator. Passing seed makes this
      function deterministic, which is needed for gradient checking but not in
      real networks.

  Outputs:
  - out: Array of the same shape as x.
  - cache: A tuple (dropout_param, mask). In training mode, mask is the dropout
    mask that was used to multiply the input; in test mode, mask is None.
  """
  p, mode = dropout_param['p'], dropout_param['mode']
  if 'seed' in dropout_param:
    np.random.seed(dropout_param['seed'])

  mask = None
  out = None

  if mode == 'train':
    ###########################################################################
    # TODO: Implement the training phase forward pass for inverted dropout.   #
    # Store the dropout mask in the mask variable.                            #
    ###########################################################################
    mask = (np.random.rand(*x.shape) > p) / (1 - p)
    out = mask * x  # Drop!
    ###########################################################################
    #                            END OF YOUR CODE                             #
    ###########################################################################
  elif mode == 'test':
    ###########################################################################
    # TODO: Implement the test phase forward pass for inverted dropout.       #
    ###########################################################################
    mask = np.ones(x.shape)
    out = x
    ###########################################################################
    #                            END OF YOUR CODE                             #
    ###########################################################################

  cache = (dropout_param, mask)
  out = out.astype(x.dtype, copy=False)

  return out, cache


def dropout_backward(dout, cache):
  """
  Perform the backward pass for (inverted) dropout.

  Inputs:
  - dout: Upstream derivatives, of any shape
  - cache: (dropout_param, mask) from dropout_forward.
  """
  dropout_param, mask = cache
  mode = dropout_param['mode']
  
  dx = None
  if mode == 'train':
    ###########################################################################
    # TODO: Implement the training phase backward pass for inverted dropout.  #
    ###########################################################################
    dx = dout * mask
    ###########################################################################
    #                            END OF YOUR CODE                             #
    ###########################################################################
  elif mode == 'test':
    dx = dout
  return dx


def conv_forward_naive(x, w, b, conv_param):
  """
  A naive implementation of the forward pass for a convolutional layer.

  The input consists of N data points, each with C channels, height H and width
  W. We convolve each input with F different filters, where each filter spans
  all C channels and has height HH and width WW.

  Input:
  - x: Input data of shape (N, C, H, W)
  - w: Filter weights of shape (F, C, HH, WW)
  - b: Biases, of shape (F,)
  - conv_param: A dictionary with the following keys:
    - 'stride': The number of pixels between adjacent receptive fields in the
      horizontal and vertical directions.
    - 'pad': The number of pixels that will be used to zero-pad the input.

  Returns a tuple of:
  - out: Output data, of shape (N, F, H', W') where H' and W' are given by
    H' = 1 + (H + 2 * pad - HH) / stride
    W' = 1 + (W + 2 * pad - WW) / stride
  - cache: (x, w, b, conv_param)
  """
  out = None
  #############################################################################
  # TODO: Implement the convolutional forward pass.                           #
  # Hint: you can use the function np.pad for padding.                        #
  #############################################################################
  N, C, H, W = x.shape
  F, C, HH, WW = w.shape
  pad = conv_param['pad']
  stride = conv_param['stride']
  Ht = 1 + (H + 2 * pad - HH) / stride
  Wt = 1 + (W + 2 * pad - WW) / stride
  # print Ht, Wt
  out = np.zeros([N, F, Ht, Wt])
  for iternum in xrange(N):  # For every single data
    data = x[iternum]
    paddata = np.zeros([C, H + 2 * pad, W + 2 * pad])
    paddata[:, pad: -pad, pad: -pad] = data
    for filternum in xrange(F): # For every single filter
      # This part code is for making the filter move to every possible position
      myfilter = w[filternum]
      # print paddata.shape
      # print myfilter.shape
      _, paddata_row, paddata_col = paddata.shape

      # This part is little annoying
      outrow = 0
      for row in xrange(0, paddata_row - HH + 1, stride):
        outcol = 0
        for col in xrange(0, paddata_col - WW + 1, stride):
          # print "row=%d, col=%d" % (row, col)
          out[iternum, filternum, outrow, outcol] = np.sum(paddata[:, row:row + HH, col:col + WW] * myfilter)
          out[iternum, filternum, outrow, outcol] += b[filternum]
          outcol += 1
        outrow += 1
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  cache = (x, w, b, conv_param)
  return out, cache


def conv_backward_naive(dout, cache):
  """
  A naive implementation of the backward pass for a convolutional layer.

  Inputs:
  - dout: Upstream derivatives.
  - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

  Returns a tuple of:
  - dx: Gradient with respect to x
  - dw: Gradient with respect to w
  - db: Gradient with respect to b
  """
  dx, dw, db = None, None, None
  #############################################################################
  # TODO: Implement the convolutional backward pass.                          #
  #############################################################################
  x, w, b, conv_param = cache
  N, C, H, W = x.shape
  F, C, HH, WW = w.shape
  pad = conv_param['pad']
  stride = conv_param['stride']
  Ht = 1 + (H + 2 * pad - HH) / stride
  Wt = 1 + (W + 2 * pad - WW) / stride

  # This part is for db
  db = np.zeros(F,)
  for i in xrange(N):
    for j in xrange(F):
      te = dout[i, j, :, :]
      db[j] += np.sum(te)

  # This is for dw, Here is what I'm thinking about: I look every point in the dout,
  # then this point can refer to some filter, then I could compute the dw

  # print dout.shape
  dw = np.zeros([F, C, HH, WW])
  paddata = np.zeros([C, H + 2 * pad, W + 2 * pad])
  for iternum in xrange(N):
    paddata[:, pad: -pad, pad: -pad] = x[iternum]
    for filternum in xrange(F):

      strow = 0
      for row in xrange(Ht):
        stcol = 0
        for col in xrange(Wt):
          filterdata = paddata[:, strow:strow + HH, stcol:stcol + WW]
          dw[filternum, :, :, :] += dout[iternum, filternum, row, col] * filterdata
          stcol += stride
        strow += stride

  # This is for dx
  dx = np.zeros([N, C, H, W])
  paddata = np.zeros([C, H + 2 * pad, W + 2 * pad])
  paddx = np.zeros([N, C, H + 2 * pad, W + 2 * pad])
  for iternum in xrange(N):
    paddata[:, pad: -pad, pad: -pad] = x[iternum]
    for filternum in xrange(F):
      strow = 0
      myfilter = w[filternum]
      for row in xrange(Ht):
        stcol = 0
        for col in xrange(Wt):
          paddx[iternum, :, strow:strow + HH, stcol:stcol + WW] += myfilter * dout[iternum, filternum, row, col]
          stcol += stride
        strow += stride
  dx = paddx[:, :, pad:-pad, pad:-pad]
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dx, dw, db


def max_pool_forward_naive(x, pool_param):
  """
  A naive implementation of the forward pass for a max pooling layer.

  Inputs:
  - x: Input data, of shape (N, C, H, W)
  - pool_param: dictionary with the following keys:
    - 'pool_height': The height of each pooling region
    - 'pool_width': The width of each pooling region
    - 'stride': The distance between adjacent pooling regions

  Returns a tuple of:
  - out: Output data
  - cache: (x, pool_param)
  """
  out = None
  #############################################################################
  # TODO: Implement the max pooling forward pass                              #
  #############################################################################
  N, C, H, W = x.shape

  pool_height = pool_param['pool_height']
  pool_width = pool_param['pool_width']
  stride = pool_param['stride']

  outWidth = (W - pool_width) / stride + 1
  outHeight = (H - pool_height) / stride + 1
  out = np.zeros([N, C, outHeight, outWidth])

  for iternum in xrange(N):
    for channel in xrange(C):
      strow = 0
      data = x[iternum, channel, :, :]
      for row in xrange(outHeight):
        stcol = 0
        for col in xrange(outWidth):
          out[iternum, channel, row, col] = np.max(data[strow:strow+pool_height, stcol:stcol+pool_width])
          stcol += stride
        strow += stride
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  cache = (x, pool_param)
  return out, cache


def max_pool_backward_naive(dout, cache):
  """
  A naive implementation of the backward pass for a max pooling layer.

  Inputs:
  - dout: Upstream derivatives
  - cache: A tuple of (x, pool_param) as in the forward pass.

  Returns:
  - dx: Gradient with respect to x
  """
  dx = None
  #############################################################################
  # TODO: Implement the max pooling backward pass                             #
  #############################################################################
  x, pool_param = cache
  N, C, H, W = x.shape

  pool_height = pool_param['pool_height']
  pool_width = pool_param['pool_width']
  stride = pool_param['stride']

  outWidth = (W - pool_width) / stride + 1
  outHeight = (H - pool_height) / stride + 1
  dx = np.zeros(x.shape)

  for iternum in xrange(N):
    for channel in xrange(C):
      strow = 0
      data = x[iternum, channel, :, :]
      for row in xrange(outHeight):
        stcol = 0
        for col in xrange(outWidth):
          te = data[strow:strow + pool_height, stcol:stcol + pool_width]
          pos = np.where(te == np.max(te))
          posrow = strow + pos[0][0]
          poscol = stcol + pos[1][0]
          dx[iternum, channel, posrow, poscol] = dout[iternum, channel, row, col]
          stcol += stride
        strow += stride
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dx


def spatial_batchnorm_forward(x, gamma, beta, bn_param):
  """
  Computes the forward pass for spatial batch normalization.
  
  Inputs:
  - x: Input data of shape (N, C, H, W)
  - gamma: Scale parameter, of shape (C,)
  - beta: Shift parameter, of shape (C,)
  - bn_param: Dictionary with the following keys:
    - mode: 'train' or 'test'; required
    - eps: Constant for numeric stability
    - momentum: Constant for running mean / variance. momentum=0 means that
      old information is discarded completely at every time step, while
      momentum=1 means that new information is never incorporated. The
      default of momentum=0.9 should work well in most situations.
    - running_mean: Array of shape (D,) giving running mean of features
    - running_var Array of shape (D,) giving running variance of features
    
  Returns a tuple of:
  - out: Output data, of shape (N, C, H, W)
  - cache: Values needed for the backward pass
  """
  out, cache = None, None

  #############################################################################
  # TODO: Implement the forward pass for spatial batch normalization.         #
  #                                                                           #
  # HINT: You can implement spatial batch normalization using the vanilla     #
  # version of batch normalization defined above. Your implementation should  #
  # be very short; ours is less than five lines.                              #
  #############################################################################
  pass
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return out, cache


def spatial_batchnorm_backward(dout, cache):
  """
  Computes the backward pass for spatial batch normalization.
  
  Inputs:
  - dout: Upstream derivatives, of shape (N, C, H, W)
  - cache: Values from the forward pass
  
  Returns a tuple of:
  - dx: Gradient with respect to inputs, of shape (N, C, H, W)
  - dgamma: Gradient with respect to scale parameter, of shape (C,)
  - dbeta: Gradient with respect to shift parameter, of shape (C,)
  """
  dx, dgamma, dbeta = None, None, None

  #############################################################################
  # TODO: Implement the backward pass for spatial batch normalization.        #
  #                                                                           #
  # HINT: You can implement spatial batch normalization using the vanilla     #
  # version of batch normalization defined above. Your implementation should  #
  # be very short; ours is less than five lines.                              #
  #############################################################################
  pass
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return dx, dgamma, dbeta
  

def svm_loss(x, y):
  """
  Computes the loss and gradient using for multiclass SVM classification.

  Inputs:
  - x: Input data, of shape (N, C) where x[i, j] is the score for the jth class
    for the ith input.
  - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
    0 <= y[i] < C

  Returns a tuple of:
  - loss: Scalar giving the loss
  - dx: Gradient of the loss with respect to x
  """
  N = x.shape[0]
  correct_class_scores = x[np.arange(N), y]
  margins = np.maximum(0, x - correct_class_scores[:, np.newaxis] + 1.0)
  margins[np.arange(N), y] = 0
  loss = np.sum(margins) / N
  num_pos = np.sum(margins > 0, axis=1)
  dx = np.zeros_like(x)
  dx[margins > 0] = 1
  dx[np.arange(N), y] -= num_pos
  dx /= N
  return loss, dx


def softmax_loss(x, y):
  """
  Computes the loss and gradient for softmax classification.

  Inputs:
  - x: Input data, of shape (N, C) where x[i, j] is the score for the jth class
    for the ith input.
  - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
    0 <= y[i] < C

  Returns a tuple of:
  - loss: Scalar giving the loss
  - dx: Gradient of the loss with respect to x
  """
  probs = np.exp(x - np.max(x, axis=1, keepdims=True))
  probs /= np.sum(probs, axis=1, keepdims=True)
  N = x.shape[0]
  loss = -np.sum(np.log(probs[np.arange(N), y])) / N
  dx = probs.copy()
  dx[np.arange(N), y] -= 1
  dx /= N
  return loss, dx
