import NNLayers
import numpy as np
import ActivationFunctions
import sklearn
import sklearn.datasets
import sklearn.linear_model
import matplotlib.pyplot as plt

np.random.seed(3) # set a seed so that the results are consistent

DNN = list()

nepochs = 1000
nsamples = 400
alpha = 1.2

####################################
def load_planar_dataset():
    np.random.seed(1)
    m = nsamples # number of examples
    N = int(m/2) # number of points per class
    D = 2 # dimensionality
    X = np.zeros((m,D)) # data matrix where each row is a single example
    Y = np.zeros((m,1), dtype='uint8') # labels vector (0 for red, 1 for blue)
    a = 4 # maximum ray of the flower

    for j in range(2):
        ix = range(N*j,N*(j+1))
        t = np.linspace(j*3.12,(j+1)*3.12,N) + np.random.randn(N)*0.2 # theta
        r = a*np.sin(4*t) + np.random.randn(N)*0.2 # radius
        X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
        Y[ix] = j
        
    X = X.T
    Y = Y.T

    return X, Y
####################################

X, Y = load_planar_dataset()

DNN.append( NNLayers.InputLayer( X ) )
DNN.append( NNLayers.FeedFwdLayer( nnodes=4, actfcn='Tanh' ) )
DNN.append( NNLayers.FeedFwdLayer( nnodes=1, actfcn='Sigmoid' ) )

nlayers = len( DNN )

CrossEntropyFcn = ActivationFunctions.CrossEntropy()

np.random.seed(2)
# initialization
for i in range( 1, nlayers ):   
    DNN[i].SetNNodesPrev( DNN[i-1].GetNNodes() )
    DNN[i].InitParams()    
    
# main loop
for j in range( nepochs ):
    
    # dummy init A
    A = None
    
    # forward prop
    for i in range( nlayers ):
        #print( i )
        A = DNN[i].ForwardProp( A )
        #print( 'A%i: ' % (i), DNN[i].A )
        #print( 'W%i: ' % (i), DNN[i].W )
        
    # compute cost
    C = -1/nsamples * np.sum( CrossEntropyFcn( Y, A ) )
    
    # initialize gradient (why needs negative gradient?)
    dA = - CrossEntropyFcn.gradient( Y, A )
    
    # backward propagation
    for i in range( nlayers-1, -1, -1 ):
        #print( i )
        dA = DNN[i].BackwardProp( dA )    
        #print( 'dZ%i: ' % (i), DNN[i].dZ )
        #print( 'dW%i: ' % (i), DNN[i].dW )
        
    # update parameters
    for i in range( nlayers ):
        #print( i )
        A = DNN[i].Update( alpha )
        
    if j % 100 == 0:
        print( '========= EPOCH %i =========' % (j) )
        print( 'Cost:', C )
    
print( 'Cost:', C )    
# predict
