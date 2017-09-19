import tensorflow as tf
import math
import numpy as np
from numpy import linalg as LA


class LookupTable(object):
    def __init__(self, m, dtype = tf.float32):
        self.cosVal = []
        self.m = m
        self.dtype = dtype
        for i in range(m):
            self.cosVal.append( math.cos(1.*i/m*math.pi) )

    def lookup(self, val):
        out = tf.zeros_like(val)
        for i in range(1,self.m):
            # 1. tf.cast
            out += tf.cast(tf.less(val, self.cosVal[i]), self.dtype)
            # 2. here maybe we can use tf.case (TODO)
            # Maybe more ... (TODO)
        return out



def Lsoftmax_loss(X, W, b, m, dtype = tf.float32):
  lookup_table = LookupTable(m, dtype)
  vec_prod = tf.matmul(X, W) + b;
  val_prod = tf.matmul(tf.expand_dims(tf.sqrt(tf.reduce_sum(tf.square(X),1)+1), 1),
             tf.expand_dims(tf.sqrt(tf.reduce_sum(tf.square(W),0)+tf.square(b)), 0) )
  cos_theta = tf.div(vec_prod, val_prod)
  k = lookup_table.lookup(cos_theta)

  # Compute cos_mtheta
  # 1. use acos
  cos_mtheta = tf.cos(m*tf.acos(cos_theta))
  # 2. use deplotment (TODO)
  # 3. use complex number (TODO)


  l_score = (tf.abs(cos_mtheta) - 2*k) * val_prod
  l_softmax = tf.nn.softmax(l_score)
  return l_softmax

if __name__ == '__main__':
  testSess = tf.Session()

  testX =  tf.constant([1, 2, 3, 4, 5, 6], dtype='float32', shape=[2, 3])
  testW =  tf.constant([3,2,6,4,9,6], dtype='float32', shape=[3,2])
  testb =  tf.constant([-1,1], dtype='float32')
  testResult = Lsoftmax_loss(testX, testW, testb, 5)

  X = np.matrix([[1,2,3,1],[4,5,6,1]])
  W = np.matrix([[3,2],[6,4],[9,6],[-1,1]])
  r = (X*W)/( np.matrix(LA.norm(X,axis=1)).T * np.matrix(LA.norm(W,axis=0)) )
  print(r)
  print(testSess.run(testResult))

  # Test LookupTable
  # testLookupTable = LookupTable(10)
  # testLookupTable2 = LookupTable(3)
  # testx =  tf.constant(np.array(range(21))*0.10-1, dtype='float32')
  # lalala = testLookupTable.lookup(testx)
  #
  #
  # print (testLookupTable.cosVal)
  # print (testSess.run(lalala))
  # print (testSess.run(testx))
