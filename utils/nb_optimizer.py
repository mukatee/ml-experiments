__author__ = 'teemu kanstren'

from hpsklearn import HyperoptEstimator, gaussian_nb
from sklearn.datasets import fetch_openml
import numpy as np

digits = fetch_openml('mnist_784')

X = digits.data
y = digits.target

test_size = int( 0.2 * len( y ) )
np.random.seed( 1 )
indices = np.random.permutation(len(X))
X_train = X[ indices[:-test_size]]
y_train = y[ indices[:-test_size]]
X_test = X[ indices[-test_size:]]
y_test = y[ indices[-test_size:]]

estim = HyperoptEstimator( classifier=gaussian_nb('mynb') )

estim.fit( X_train, y_train )

print( estim.score( X_test, y_test ) )
# <<show score here>>
print( estim.best_model() )
# <<show model here>>
pass
