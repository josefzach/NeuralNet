import NNLayers
import numpy as np
import ActivationFunctions
DNN = list()

nepochs = 2
nsamples = 100
alpha = 0.1

DNN.append( NNLayers.InputLayer( np.random.randn( 50, nsamples ) ) )
DNN.append( NNLayers.FeedFwdLayer( nnodes=10, actfcn='ReLU' ) )
DNN.append( NNLayers.FeedFwdLayer( nnodes=5, actfcn='ReLU' ) )
DNN.append( NNLayers.FeedFwdLayer( nnodes=1, actfcn='Sigmoid' ) )

nlayers = len( DNN )

CrossEntropyFcn = ActivationFunctions.CrossEntropy()
Y = np.random.randn( 1, nsamples )

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
        
    # compute cost
    # TODO
    
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
        print( j )