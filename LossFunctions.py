import numpy as np
#import warnings
#warnings.filterwarnings("error")

def Factory( lossfcn ):
    if lossfcn == 'CrossEntropy':
        return CrossEntropy()
    else:
        return None

class CrossEntropy:
    def __init__( self ):
        self.eps = 1e-12

    def __call__( self, y, a ):
        try:
            y = np.multiply( np.log( a + self.eps ), y ) + np.multiply( np.log( 1 - a + self.eps ), 1-y )
        except RuntimeWarning:
            print( 'a was:', a )
        return y
    
    def gradient( self, y, a ):
        try:
            y = np.multiply( 1/(a + self.eps), y ) - np.multiply( 1/( 1 - a + self.eps), 1-y )
        except RuntimeWarning:
            print( 'a was:', a )
        return y
