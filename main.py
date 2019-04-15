import NNLayers
import numpy as np
import ActivationFunctions
DNN = list()

nepochs = 10000
nsamples = 1000
alpha = 0.1

X_train = np.random.uniform( -1, 1, (1, nsamples) )
Y_train = ( X_train >= 0 )

DNN.append( NNLayers.InputLayer( X_train ) )
DNN.append( NNLayers.FeedFwdLayer( nnodes=10, actfcn='ReLU' ) )
DNN.append( NNLayers.FeedFwdLayer( nnodes=5, actfcn='ReLU' ) )
DNN.append( NNLayers.FeedFwdLayer( nnodes=1, actfcn='Sigmoid' ) )

nlayers = len( DNN )

CrossEntropyFcn = ActivationFunctions.CrossEntropy()
Y = Y_train

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
        #print( 'Forward prop Z: ', DNN[i].W )
        
    # compute cost
    C = -1/nsamples * np.sum( A )
    
    # initialize gradient
    dA = CrossEntropyFcn.gradient( Y, A )
    
    # backward propagation
    for i in range( nlayers-1, -1, -1 ):
        #print( i )
        dA = DNN[i].BackwardProp( dA )    
    
    # update parameters
    for i in range( nlayers ):
        #print( i )
        A = DNN[i].Update( alpha )
        
    if j % 100 == 0:
        print( '========= EPOCH %i =========' % (j) )
        print( 'Cost:', C )
        
# predict
