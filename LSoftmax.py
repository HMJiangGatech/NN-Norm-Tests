import tensorflow as tf
import math
import numpy as np
from numpy import linalg as LA


class LookupTable(object):
    cosVal = []
    m = 0
    def __init__(self, m):
        self.m = m
        for i in range(m):
            self.cosVal.append( math.cos(1.*i/m*math.pi) )

    def lookup(self, val):
        for i in range(1,m+1):
            if self.cosVal[i] <= val:
                return i-1



def Lsoftmax_loss(X, W, b, m):
  lookup_table = LookupTable(m)
  vec_prod = tf.matmul(X, W) + b;
  val_prod = tf.matmul(tf.expand_dims(tf.sqrt(tf.reduce_sum(tf.square(X),1)+1), 1),
             tf.expand_dims(tf.sqrt(tf.reduce_sum(tf.square(W),0)+tf.square(b)), 0) )
  cos_theta = tf.div(vec_prod, val_prod)
  k = lookup_table.lookup(cos_theta)
  cos_ntheta = tf.cos(m*tf.acos(cos_theta))
  l_score = tf.abs(cos_ntheta)
  return cos_theta

if __name__ == '__main__':
  testX =  tf.constant([1, 2, 3, 4, 5, 6], dtype='float32', shape=[2, 3])
  testW =  tf.constant([3,2,6,4,9,6], dtype='float32', shape=[3,2])
  testb =  tf.constant([-1,1], dtype='float32')
  testResult = Lsoftmax_loss(testX, testW, testb, 1)
  testSess = tf.Session()

  X = np.matrix([[1,2,3,1],[4,5,6,1]])
  W = np.matrix([[3,2],[6,4],[9,6],[-1,1]])
  r = (X*W)/( np.matrix(LA.norm(X,axis=1)).T * np.matrix(LA.norm(W,axis=0)) )
  print(r)
  print (testSess.run(testResult))
