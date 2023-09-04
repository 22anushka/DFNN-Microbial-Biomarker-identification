```
@author: Anushka S, Syama K
```

import numpy as np
import tensorflow as tf
from keras.layers import Dense

class MyConstraint(tf.keras.constraints.Constraint):

  def __init__(self, my_constraints):
    self.my_constraints = my_constraints

  def __call__(self, w):
    assert tf.shape(w).shape == tf.shape(self.my_constraints).shape
    
    factor = tf.cast(self.my_constraints, dtype=tf.float32)
    return w*factor

  def get_config(self):
    return {'my_constraints': self.my_constraints}

def Sparselayer(adjacency_mat, kernelInitializer, kernelConstraint, nameOfLayer, inp):
  my_constraints = tf.constant(adjacency_mat, dtype=tf.float32)
  
  if kernelConstraint==1:
	  return Dense(units=np.shape(adjacency_mat)[1], activation='relu', name=nameOfLayer, use_bias=True, kernel_initializer=kernelInitializer, kernel_constraint=MyConstraint(my_constraints))(inp)
  else:
    return Dense(units=np.shape(adjacency_mat)[1], activation='relu', name=nameOfLayer, use_bias=True, kernel_initializer=kernelInitializer)(inp)