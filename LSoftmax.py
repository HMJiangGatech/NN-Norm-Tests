import tensorflow as tf
import math
import numpy as np
from scipy.special import comb
from numpy import linalg as LA


class LookupTable(object):
    def __init__(self, m, dtype = tf.float32):
        self.cosVal = []
        self.m = m
        self.dtype = dtype
        for i in range(m+1):
            self.cosVal.append( math.cos(1.*i/m*math.pi) )

    def lookup(self, val):
        # 1. tf.cast
        out = tf.zeros_like(val)
        for i in range(1,self.m):
            out += tf.cast(tf.less(val, self.cosVal[i]), self.dtype)

        # 2. here maybe we can use tf.case (TODO: seeeems doesn't work)
        # mdict = {}
        # funct = {0: lambda: tf.constant(0)}
        # for i in range(1,self.m):
        #     funct[i] = lambda: tf.constant(i)
        #     mdict[tf.logical_and(tf.less(val, self.cosVal[i]),
        #           tf.greater_equal(val, self.cosVal[i+1]))] = funct[i]
        # out = tf.case(mdict,default=funct[0])
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
  # cos_mtheta = tf.cos(m*tf.acos(cos_theta))

  # 2. use deployment
  sin2_theta = 1 - cos_theta**2
  cos_mtheta = tf.zeros_like(cos_theta)
  for i in range(0,m+1,2):
      cos_mtheta += (-1)**(i//2) * comb(m,i) \
        * cos_theta**(m-i) * sin2_theta**(i//2)

  # 3. use complex number (TODO)


  l_score = ( tf.multiply(tf.pow(-1.,k), cos_mtheta) - 2*k) * val_prod
  return l_score

if __name__ == '__main__':
  testSess = tf.Session()

  m = 3

  X_seed = np.random.rand(2*3) - 0.5
  W_seed = np.random.rand(3*2) - 0.5
  b_seed = np.random.rand(2) - 0.5

  testX =  tf.constant(X_seed, dtype='float32', shape=[2,3])
  testW =  tf.constant(W_seed, dtype='float32', shape=[3,2])
  testb =  tf.constant(b_seed, dtype='float32')
  testResult = Lsoftmax_loss(testX, testW, testb, m)

  X = np.matrix( np.concatenate((X_seed.reshape((2,3)), [[1],[1]]), axis=1)  )
  W = np.matrix( np.concatenate((W_seed.reshape((3,2)), [b_seed]), axis=0) )
  val_prod = ( np.matrix(LA.norm(X,axis=1)).T * np.matrix(LA.norm(W,axis=0)) )
  r = (X*W)/val_prod

  cos_mx = np.cos( m * np.arccos(r) )

  k = r*0
  for i in range(1,m):
    k += (r<math.cos(1.0*i*math.pi/m))

  l_score = np.multiply(np.power(-1, k) , cos_mx)
  l_score = (l_score  - 2*k)
  l_score = np.multiply(l_score, val_prod)

  print(l_score)
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
