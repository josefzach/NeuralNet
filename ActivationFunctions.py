import numpy as np

def Factory( actfcn ):
    if actfcn == 'Sigmoid':
        return Sigmoid()
    elif actfcn == 'ReLU':
        return ReLU()
    else:
        return None
        
class Sigmoid:
    def __call__( self, x ):
        y = 1 / ( 1 + np.exp( -x ) )
        return y
    
    def gradient( self, x ):
        #print( 'Sigmoid gradient input: %f' % (x) )
        func = self( x )
        y = np.multiply( func, ( 1 - func ) )
        return y
        
class ReLU:
    def __call__( self, x ):
        y = x
        y[ x <= 0 ] = 0
        return y
    
    def gradient( self, x ):
        y = np.ones( np.shape( x ) )
        y[ x <= 0 ] = 0
        return y
        
class CrossEntropy:
    def __call__( self, y, yest ):
        y = np.multiply( np.log( yest ), y ) + np.multiply( np.log( 1 - yest ), 1-y )
        return y
        #cost = -1/m * np.sum( logprobs )
    
    def gradient( self, y, yest ):
        y = np.multiply( 1/yest, y ) - np.multiply( 1/( 1 - yest ), 1-y )
        return y