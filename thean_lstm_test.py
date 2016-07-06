
import numpy
import theano
from theano import config
import theano.tensor as tensor
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

import thean_lstm_lmbd as imdb


testdatasets = {'imdb': (imdb.load_data(path='imdb.pkl'), imdb.prepare_data)}