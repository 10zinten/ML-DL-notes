'''
- We're going to use logistic regression to predict if someone has diabetes or not given 3 body metrics! We'll use Newton's
  method to help us optimize the model.
- We'll use a bit from all the mathematical disciplines i've mentioned (calculus, linear algebra, probability theory,
  statistics)
'''

#matrix math
import numpy as np
#data manipulation
import pandas as pd
#matirx data structure
from patsy import dmatrices
#for error logging
import warnings


#outputs probability btw 0 and 1, used to help define our logistic regression curve
def sigmid(x):
  '''Sigmod function of x.'''
  return 1/(1+np.exp(-x))

np.random.seed(0) # set the seed.



