import numpy as np
#import warnings
#warnings.filterwarnings("error")

def Factory( lossfcn ):
    if lossfcn == 'CrossEntropy':
        return CrossEntropy()
    else:
        return None

class CrossEntropy:
    def __call__( self, y, a ):
        y = np.multiply( np.log( a ), y ) + np.multiply( np.log( 1 - a ), 1-y )
        return y
        #cost = -1/m * np.sum( logprobs )
    
    def gradient( self, y, a ):
        try:
            y = np.multiply( 1/a, y ) - np.multiply( 1/( 1 - a ), 1-y )
        except RuntimeWarning:
            print( 'a was:', a )
        return y
