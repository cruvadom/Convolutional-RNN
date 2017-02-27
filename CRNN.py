import tensorflow as tf

r"""Performs the 1-D Convolutional RNN Operation, according to the paper:
  Convolutional RNN: an Enhanced Model for Extracting Features from Sequential Data (https://arxiv.org/abs/1602.05875)
  Gil Keren and Bjoern Schuller. 

  Calling the below function is equivalnet to applying one CRNN layer. For a deep model with a few
  CRNN layers, the function should be invoked multiple times. 

  Given a tensor, the function extracts patches of `kernel_size` time-steps, and processed each 
  with one or more recurrent layers. The hidden state of the recurrent neural network is then 
  returned as the feature vector representing the path. 

  Args:
    tensor: The tensor to perform the operation on, shape `[batch, time-steps, features]`
            or `[batch, time-steps, features, 1]`.
    kernel_size: The number of time-steps to include in every patch/window (same as in standard 1-D convolution).
    stride: the number of time-steps between two consecutive patches/windows (same as in standard 1-D convolution).
    out_channels: The number of extracted features from each patch/window (in standard 1-D convolution 
                  known as the number of feature maps), which is the hidden dimension of the recurrent 
                  layers that processes each patch/window.
    rnn_n_layers: The number of recurrent layers to process the patches/windows. 
		  (in the original paper was always =1). 
    rnn_type: Type of recurrent layers to use: `simple`/`lstm`/`gru`
    bidirectional: Whether to use a bidirectional recurrent layers (such as BLSTM, when the rnn_type is 'lstm'). 
                   If True, The actual number of extracted features from each patch/window is `2 * out_channels`.
    w_std: Weights in the recurrent layers will be initialized randomly using a Gaussaian distribution with
           zero mean and a standard deviation of `w_std`. Biases are initialized with zero. 
    padding: `SAME` or `VALID` (same as in standard 1-D convolution).
    scope_name: For variable naming, the name prefix for variables names.  

  Returns:
    A 3-D `Tensor` with shape `[batch, time-steps, features]`, similarly to the output of a standard 1-D convolution. 
  """
def crnn(tensor, kernel_size, stride, out_channels, rnn_n_layers, rnn_type, bidirectional, w_std, padding, scope_name):
  with tf.variable_scope(scope_name, initializer=tf.truncated_normal_initializer(stddev=w_std)):
    # Expand to have 4 dimensions if needed
    if len(tensor.shape) == 3:
      tensor = tf.expand_dims(tensor, 3)
    
    # Extract the patches (returns [batch, time-steps, 1, patch content flattened])
    batch_size = tensor.shape[0].value
    n_in_features = tensor.shape[2].value
    patches = tf.extract_image_patches(images=tensor, 
                             ksizes=[1, kernel_size, n_in_features, 1], 
                             strides=[1, stride, n_in_features, 1], 
                             rates=[1, 1, 1, 1], 
                             padding=padding)
    patches = patches[:, :, 0, :]
    
    # Reshape to do: 
    # 1) reshape the flattened patches back to [kernel_size, n_in_features]
    # 2) combine the batch and time-steps dimensions (which will be the new 'batch' size, for the RNN)
    # now shape will be [batch * time-steps, kernel_size, n_features]
    time_steps_after_stride = patches.shape[1].value
    patches = tf.reshape(patches, [batch_size * time_steps_after_stride, kernel_size, n_in_features])
    
    # Transpose and convert to a list, to fit the tf.contrib.rnn.static_rnn requirements
    # Now will be a list of length kernel_size, each element of shape [batch * time-steps, n_features]
    patches = tf.unstack(tf.transpose(patches, [1, 0, 2]))
    
    # Create the RNN Cell
    if rnn_type == 'simple':
      rnn_cell_func = tf.contrib.rnn.BasicRNNCell
    elif rnn_type == 'lstm':
      rnn_cell_func = tf.contrib.rnn.LSTMBlockCell
    elif rnn_type == 'gru':
      rnn_cell_func = tf.contrib.rnn.GRUBlockCell
    if not bidirectional:
      rnn_cell = rnn_cell_func(out_channels)
    else:
      rnn_cell_f = rnn_cell_func(out_channels)
      rnn_cell_b = rnn_cell_func(out_channels)
      
    # Multilayer RNN? (does not appear in the original paper)
    if rnn_n_layers > 1:
      if not bidirectional:
        rnn_cell = tf.contrib.rnn.MultiRNNCell([rnn_cell] * rnn_n_layers)
      else:
        rnn_cell_f = tf.contrib.rnn.MultiRNNCell([rnn_cell_f] * rnn_n_layers)
        rnn_cell_b = tf.contrib.rnn.MultiRNNCell([rnn_cell_b] * rnn_n_layers)
    
    # The RNN itself
    if not bidirectional:
      outputs, state = tf.contrib.rnn.static_rnn(rnn_cell, patches, dtype=tf.float32)
    else:
      outputs, output_state_fw, output_state_bw = tf.contrib.rnn.static_bidirectional_rnn(rnn_cell_f, rnn_cell_b, patches, dtype=tf.float32)
    
    # Use only the output of the last time-step (shape will be [batch * time-steps, out_channels]).
    # In the case of a bidirectional RNN, we want to take the last time-step of the forward RNN, 
    # and the first time-step of the backward RNN. 
    if not bidirectional:
      outputs = outputs[-1]
    else:
      half = int(outputs[0].shape.as_list()[-1] / 2)
      outputs = tf.concat([outputs[-1][:,:half], 
                           outputs[0][:,half:]], 
                          axis=1)
    
    # Expand the batch * time-steps back (shape will be [batch_size, time_steps, out_channels]
    if bidirectional:
      out_channels = 2 * out_channels
    outputs = tf.reshape(outputs, [batch_size, time_steps_after_stride, out_channels])
    
    return outputs
      
