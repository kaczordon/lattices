# Copyright 2015 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Example / benchmark for building a PTB LSTM model.

Trains the model described in:
(Zaremba, et. al.) Recurrent Neural Network Regularization
http://arxiv.org/abs/1409.2329

There are 3 supported model configurations:
===========================================
| config | epochs | train | valid  | test
===========================================
| small  | 13     | 37.99 | 121.39 | 115.91
| medium | 39     | 48.45 |  86.16 |  82.07
| large  | 55     | 37.87 |  82.62 |  78.29
The exact results may vary depending on the random initialization.

The hyperparameters used in the model:
- init_scale - the initial scale of the weights
- learning_rate - the initial value of the learning rate
- max_grad_norm - the maximum permissible norm of the gradient
- num_layers - the number of LSTM layers
- num_steps - the number of unrolled steps of LSTM
- hidden_size - the number of LSTM units
- max_epoch - the number of epochs trained with the initial learning rate
- max_max_epoch - the total number of epochs for training
- keep_prob - the probability of keeping weights in the dropout layer
- lr_decay - the decay of the learning rate for each epoch after "max_epoch"
- batch_size - the batch size

The data required for this example is in the data/ dir of the
PTB dataset from Tomas Mikolov's webpage:

$ wget http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz
$ tar xvf simple-examples.tgz

To run:

$ python ptb_word_lm.py --data_path=simple-examples/data/

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import math
import numpy as np
import tensorflow as tf
import collections
from collections import deque
from tensorflow.models.rnn.ptb import reader

class node:
  def __init__(self, name, probability, timeS, timeE, word, path):
    self.name = name
    self.probability = probability
    self.timeS = timeS
    self.timeE = timeE
    self.word = word
    self.path = path
class pathProb:
  def __init__(self, currentProb, path, state, rnn_prob):
    self.currentProb = currentProb
    self.path = path
    self.state = state
    self.rnn_prob = rnn_prob

def read(filename):                                #returns lines of file
  with open(filename) as f:
    lines = [line.split(' ') for line in open(filename)]
    return lines

def buildwords(lines, vocab):                      #gets the word id's that this model is trained on
  words = ""
  for i in range(len(lines)):
    nextW = lines[i][5].rstrip()
    if nextW != ".sil":
      if nextW in vocab:
        words += nextW + " "
      else:
        words += "<unk>" + " "
  return words
  
def fill_map(lines, initial_state, rnn_prob):                                         #fills the adjacency map and visited map
  tmp = node(lines[0][0], -float(lines[0][2]), int(lines[0][3]), int(lines[0][4]), lines[0][5].rstrip(), []) #source node
  firstPath = pathProb(0, [lines[0][0]], initial_state, rnn_prob)                                            #sets first state
  tmp.path.append(firstPath)
  adjacency = {lines[0][0]: [tmp]}
  visited = {lines[0][0]: True}
  for i in range(len(lines)):
    if adjacency.has_key(lines[i][0]) == False:                                       #checks if key has been created for this node
      visited[lines[i][0]] = False
      adjacency[lines[i][0]] = []
      prob = -float(lines[i][2])
      ts = int(lines[i][3])
      te = int(lines[i][4])
      newNode = node(lines[i][0], 0, ts, te, lines[i][5].rstrip(), [])
      adjacency[lines[i][0]].append(newNode)
      nextNode = node(lines[i][1], prob, ts, te, lines[i][5].rstrip(), [])
      adjacency[lines[i][0]].append(nextNode)
      if i == len(lines) - 1:
        adjacency[lines[i][1]] = []
        lastNode = node(lines[i][1], 0, ts, te, lines[i][5].rstrip(), [])
        adjacency[lines[i][1]].append(lastNode)
        visited[lines[i][1]] = False
        finalNode = lines[i][1]
    else:                                                                             #the key exists and this adds another connected node
      prob = -float(lines[i][2])
      ts = int(lines[i][3])
      te = int(lines[i][4])
      nextNode = node(lines[i][1], prob, ts, te, lines[i][5].rstrip(), [])
      adjacency[lines[i][0]].append(nextNode)
  return (adjacency, visited, finalNode)

def get_rnn_prob(state, word, session, mtest):                                        #gets next state and probability layout
  prob, return_state = run_epoch(session, mtest, [word,1], tf.no_op(), state, verbose=True, prev_state=True)
  return (return_state, prob)


def checkPaths(path1, path2):                         #checks if two paths are the same
  i = 0
  while i < len(path1):
    if path1[i] != path2[i]:
      return False
    i += 1
  return True

def purgePaths(path):                                 #removes duplicate paths
  i = 0
  j = 1
  while i < len(path):
    while j < len(path):
      if len(path[i].path) == len(path[j].path):
        if checkPaths(path[i].path, path[j].path):
          del path[j]                                 #so that we don't overwrite old slice with wrong path
      j += 1
    i += 1
    j = i + 1
  return path

def traverse_lattice(adjacency, visited, source, finalNode, session, mtest, word2embedding, rw, lw):
  queue = deque([source])
  while len(queue) > 0:
    nextNode = queue.popleft()
    adjacency[nextNode].sort(key=lambda x: x.timeS)    #sorts by start time
    visited[nextNode] = False
    adjacency[nextNode][0].path = purgePaths(adjacency[nextNode][0].path) #remove duplicate paths
    for value in adjacency[nextNode]:
      if nextNode == value.name:                       #this is the root node it stores all paths coming into it
        if len(value.path) >= 5:
          topFive = []
          value.path.sort(key=lambda x: x.currentProb) #take top 5 paths
          for i in range(5):
            topFive.append(value.path[i])
          value.path = topFive
        continue
      if visited[value.name] == False:
        visited[value.name] = True
        queue.append(value.name)
      for paths in adjacency[nextNode][0].path:        #build paths to next node from previous paths
        nextPath = []
        nextPath.extend(paths.path)                    #add old paths
        nextPath.append(value.name)                    #add new node name
        rnn_word_prob = 0
        state = paths.state
        rnn_prob = paths.rnn_prob
        if value.word != ".sil":                       #check for silence
          if value.word in word2embedding:
            rnn_word_prob = paths.rnn_prob[0][word2embedding[value.word]]  #get probability of word given the current state
            state, rnn_prob = get_rnn_prob(paths.state, word2embedding[value.word], session, mtest) #get next state and probability layout
          else:
            rnn_word_prob = paths.rnn_prob[0][word2embedding["<unk>"]]     #unknown word
            state, rnn_prob = get_rnn_prob(paths.state, word2embedding["<unk>"], session, mtest)
        if rnn_word_prob != 0:                                             #sanity check
          next_rnn_prob = math.log(rnn_word_prob) * 10
        else:
          next_rnn_prob = 0
        newPath = pathProb(value.probability*lw + paths.currentProb + (-next_rnn_prob)*rw, nextPath, state, rnn_prob)
        adjacency[value.name][0].path.append(newPath)
  return adjacency[finalNode]

flags = tf.flags
logging = tf.logging

flags.DEFINE_string(
    "model", "small",
    "A type of model. Possible options are: small, medium, large.")
flags.DEFINE_string("data_path", None, "data_path")

FLAGS = flags.FLAGS


class PTBModel(object):
  """The PTB model."""
  
  def __init__(self, is_training, config):
    self.batch_size = batch_size = config.batch_size
    self.num_steps = num_steps = config.num_steps
    size = config.hidden_size
    vocab_size = config.vocab_size

    self._input_data = tf.placeholder(tf.int32, [batch_size, num_steps])
    self._targets = tf.placeholder(tf.int32, [batch_size, num_steps])

    # Slightly better results can be obtained with forget gate biases
    # initialized to 1 but the hyperparameters of the model would need to be
    # different than reported in the paper.
    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(size, forget_bias=0.0)
    if is_training and config.keep_prob < 1:
      lstm_cell = tf.nn.rnn_cell.DropoutWrapper(
          lstm_cell, output_keep_prob=config.keep_prob)
    cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * config.num_layers)

    self._initial_state = cell.zero_state(batch_size, tf.float32)

    with tf.device("/cpu:0"):
      embedding = tf.get_variable("embedding", [vocab_size, size])
      inputs = tf.nn.embedding_lookup(embedding, self._input_data)

    if is_training and config.keep_prob < 1:
      inputs = tf.nn.dropout(inputs, config.keep_prob)

    # Simplified version of tensorflow.models.rnn.rnn.py's rnn().
    # This builds an unrolled LSTM for tutorial purposes only.
    # In general, use the rnn() or state_saving_rnn() from rnn.py.
    #
    # The alternative version of the code below is:
    #
    # from tensorflow.models.rnn import rnn
    # inputs = [tf.squeeze(input_, [1])
    #           for input_ in tf.split(1, num_steps, inputs)]
    # outputs, state = rnn.rnn(cell, inputs, initial_state=self._initial_state)
    outputs = []
    state = self._initial_state
    with tf.variable_scope("RNN"):
      for time_step in range(num_steps):
        if time_step > 0: tf.get_variable_scope().reuse_variables()
        (cell_output, state) = cell(inputs[:, time_step, :], state)
        outputs.append(cell_output)

    output = tf.reshape(tf.concat(1, outputs), [-1, size])
    softmax_w = tf.get_variable("softmax_w", [size, vocab_size])
    softmax_b = tf.get_variable("softmax_b", [vocab_size])
    logits = tf.matmul(output, softmax_w) + softmax_b
    loss = tf.nn.seq2seq.sequence_loss_by_example(
        [logits],
        [tf.reshape(self._targets, [-1])],
        [tf.ones([batch_size * num_steps])])
    self._cost = cost = tf.reduce_sum(loss) / batch_size
    self._final_state = state

    if not is_training:
      self._proba = tf.nn.softmax(logits)
      return

    self._lr = tf.Variable(0.0, trainable=False)
    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars),
                                      config.max_grad_norm)
    optimizer = tf.train.GradientDescentOptimizer(self.lr)
    self._train_op = optimizer.apply_gradients(zip(grads, tvars))

  def assign_lr(self, session, lr_value):
    session.run(tf.assign(self.lr, lr_value))

  @property
  def input_data(self):
    return self._input_data

  @property
  def targets(self):
    return self._targets

  @property
  def initial_state(self):
    return self._initial_state

  @property
  def cost(self):
    return self._cost

  @property
  def final_state(self):
    return self._final_state

  @property
  def lr(self):
    return self._lr

  @property
  def train_op(self):
    return self._train_op

  @property
  def proba(self):
    return self._proba


class SmallConfig(object):
  """Small config."""
  init_scale = 0.1
  learning_rate = 1.0
  max_grad_norm = 5
  num_layers = 2
  num_steps = 20
  hidden_size = 200
  max_epoch = 4
  max_max_epoch = 13
  keep_prob = 1.0
  lr_decay = 0.5
  batch_size = 20
  vocab_size = 10000


class MediumConfig(object):
  """Medium config."""
  init_scale = 0.05
  learning_rate = 1.0
  max_grad_norm = 5
  num_layers = 2
  num_steps = 35
  hidden_size = 650
  max_epoch = 6
  max_max_epoch = 39
  keep_prob = 0.5
  lr_decay = 0.8
  batch_size = 20
  vocab_size = 10000


class LargeConfig(object):
  """Large config."""
  init_scale = 0.04
  learning_rate = 1.0
  max_grad_norm = 10
  num_layers = 2
  num_steps = 35
  hidden_size = 1500
  max_epoch = 14
  max_max_epoch = 55
  keep_prob = 0.35
  lr_decay = 1 / 1.15
  batch_size = 20
  vocab_size = 10000


class TestConfig(object):
  """Tiny config, for testing."""
  init_scale = 0.1
  learning_rate = 1.0
  max_grad_norm = 1
  num_layers = 1
  num_steps = 2
  hidden_size = 2
  max_epoch = 1
  max_max_epoch = 1
  keep_prob = 1.0
  lr_decay = 0.5
  batch_size = 20
  vocab_size = 10000


def run_epoch(session, m, data, eval_op, Pstate=None, verbose=False, prev_state=False):
  """Runs the model on the given data."""
  epoch_size = ((len(data) // m.batch_size) - 1) // m.num_steps
  start_time = time.time()
  costs = 0.0
  iters = 0
  if not prev_state:
    state = m.initial_state.eval()
  else:
    state = Pstate    #previous state
    
  for step, (x, y) in enumerate(reader.ptb_iterator(data, m.batch_size,
                                                    m.num_steps)):
    cost, state, proba, _  = session.run([m.cost, m.final_state, m.proba, eval_op],
                                 {m.input_data: x,
                                  m.targets: y,
                                  m.initial_state: state})
    costs += cost
    iters += m.num_steps
    if prev_state:   #if previous state exists only calculate for one step
      return (proba, state)
    #if verbose:# and step % (epoch_size // 10) == 10:
      #print("%.3f perplexity: %.3f speed: %.0f wps" %
      #      (step * 1.0 / epoch_size, np.exp(costs / iters),
      #       iters * m.batch_size / (time.time() - start_time)))
      #print("probability:%.18f  " % np.amax(proba))
      #index = np.argmax(proba)
      #print("word: ", index)
      #print("cost: ", cost)

  return np.exp(costs / iters)
def build_string(string):
  count = collections.Counter(string)
  count_pairs = sorted(count.items(), key=lambda x: (-x[1], x[0]))

  words,_ = list(zip(*count_pairs))
  word_to_id = dict(zip(words, range(len(words))))
  return [word_to_id[word] for word in string] 

def get_config():
  if FLAGS.model == "small":
    return SmallConfig()
  elif FLAGS.model == "medium":
    return MediumConfig()
  elif FLAGS.model == "large":
    return LargeConfig()
  elif FLAGS.model == "test":
    return TestConfig()
  else:
    raise ValueError("Invalid model: %s", FLAGS.model)


def main(_):
  if not FLAGS.data_path:
    raise ValueError("Must set --data_path to PTB data directory")

 # raw_data = reader.ptb_raw_data(FLAGS.data_path)
 # train_data, valid_data, test_data, _  = raw_data
  vocab = reader._build_vocab(FLAGS.data_path + "/ptb.train.txt") #word id's of model vocabulary
  inv_map = {v: k for k, v in vocab.items()}                      #word id's to literals
 #word_id = reader._build_vocab("test1.txt")

  config = get_config()
  eval_config = get_config()
  eval_config.batch_size = 1
  eval_config.num_steps = 1

  with tf.Graph().as_default(), tf.Session() as session:
    initializer = tf.random_uniform_initializer(-config.init_scale,
                                                config.init_scale)
    with tf.variable_scope("model", reuse=None, initializer=initializer):
      m = PTBModel(is_training=True, config=config)
    with tf.variable_scope("model", reuse=True, initializer=initializer):
      mvalid = PTBModel(is_training=False, config=config)
      mtest = PTBModel(is_training=False, config=eval_config)

    tf.initialize_all_variables().run()
    saver = tf.train.Saver()
   # for i in range(config.max_max_epoch):                                                 #training not needed
   #   lr_decay = config.lr_decay ** max(i - config.max_epoch, 0.0)
   #   m.assign_lr(session, config.learning_rate * lr_decay)

   #   print("Epoch: %d Learning rate: %.3f" % (i + 1, session.run(m.lr)))
   #   train_perplexity = run_epoch(session, m, train_data, m.train_op,
   #                                verbose=True)
   #   print("Epoch: %d Train Perplexity: %.3f" % (i + 1, train_perplexity))
   #   valid_perplexity = run_epoch(session, mvalid, valid_data, tf.no_op())
   #   print("Epoch: %d Valid Perplexity: %.3f" % (i + 1, valid_perplexity))
   #   save_path = saver.save(session, "tmp/model.epoch.%03d.ckpt" % (i + 1) )
   #   print("Model saved in file: %s" % save_path)
   
    ckpt = tf.train.get_checkpoint_state("./tmp")#,latest_filename="model.epoch.013.ckpt") #gets the model and trained weights
    if ckpt and ckpt.model_checkpoint_path:                                                #loads the model and weights
      saver.restore(session, ckpt.model_checkpoint_path)
    else:
      print( "error")                                                                      #something went wrong
    proba, state = run_epoch(session, mtest,[1,1], tf.no_op(), mtest.initial_state.eval(), verbose=True, prev_state=True)
    edges = read("/data1_home/jkawalec/lattices/00ac0u0h.wav.lattice")
    source = edges[0][0]                                                                   #start node of rescoring

     #adjacency: adjacency map of nodes in lattice
     #visited: map of booleans of visited nodes
     #source: start node
     #finalNode: end node
     #session: Tensorflow session variable
     #mtest: Model that is used to test the output, can be used to get output from trained model
     #vocab: map of words to id's
     #rw: RNN probability weight
     #lw: Lattice probability weight
    
    adjacency, visited, finalNode = fill_map(edges, mtest.initial_state.eval(), proba)     #build the adjacency map
    
    rw = 1
    lw = 1
    final = traverse_lattice(adjacency, visited, source, finalNode, session, mtest, vocab, rw, lw) 
    final[0].path.sort(key=lambda x: x.currentProb)
    for i in range(5):
          print (final[0].path[i].path, '\n')
          print(final[0].path[i].currentProb)

if __name__ == "__main__":
  tf.app.run()

  
