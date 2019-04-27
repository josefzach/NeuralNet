import numpy as np
#import warnings
#warnings.filterwarnings("error")

def Factory( actfcn ):
    if actfcn == 'Sigmoid':
        return Sigmoid()
    elif actfcn == 'ReLU':
        return ReLU()
    elif actfcn == 'SaLU':
        return SaLU()
    elif actfcn == 'Tanh':
        return Tanh()
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

class SaLU:
    def __call__( self, x ):
        y = x
        y[ x > 1 ] = 1
        y[ x < -1 ] = -1
        return y
    
    def gradient( self, x ):
        y = np.ones( np.shape( x ) )
        y[ x > 1 ] = 0
        y[ x < -1 ] = 0
        return y

class Tanh:
    def __call__( self, x ):
        y = ( np.exp( x ) - np.exp( -x ) ) / ( np.exp( x ) + np.exp( -x ) )
        return y
    
    def gradient( self, x ):
        #print( 'Tanh gradient input: %f' % (x) )
        func = self( x )
        y = 1 - np.power( func, 2 )
        return y
