import numpy as np
import ActivationFunctions as Activation

class NNLayer:
    def ForwardProp( self, X ):
        raise NotImplementedError( 'Class %s does not implement method ForwardProp()' % (self.__class__.__name__) )
    def BackwardProp( self, dX ):
        raise NotImplementedError( 'Class %s does not implement method BackwardProp()' % (self.__class__.__name__) )

class FeedFwdLayer( NNLayer ):
    def __init__( self, nnodes = 5, actfcn = 'ReLU' ):
        self.nnodes = nnodes
        self.nnodes_prev = None
        self.nsamples = None
        self.actfcn = Activation.Factory( actfcn )
        self.W = None
        self.b = None
        self.X = None
        self.Z = None
        self.A = None
        self.dA = None
        self.A_prev = None
        self.dA_prev = None
        
    def __add__( self, other ):
        print( 'SELF:', self ), print( 'OTHER:', other )
        
        # pass number of nodes on layer n to layer n+1
        other.SetNNodesPrev( self.GetNNodes() )
        other.InitParams()
        
        # pass on second operand as first argument in next iteration
        return other
    
    def GetNNodes( self ):
        return self.nnodes
    
    def SetNNodesPrev( self, nnodes_prev ):
        self.nnodes_prev = nnodes_prev
    
    def InitParams( self ):
        self.W = np.random.randn( self.nnodes,self.nnodes_prev ) * 0.01
        self.b = np.zeros( ( self.nnodes, 1 ) )
        
    def ForwardProp( self, X ):
        self.nsamples = np.shape( X )[1]
        self.A_prev = X
        self.Z = np.dot( self.W, X ) + self.b
        self.A = self.actfcn( self.Z )
        return self.A
        
    def BackwardProp( self, dA ):
        self.dA = dA
        self.dZ = np.multiply( self.dA, self.actfcn.gradient( self.Z ) )
        self.dW = 1 / self.nsamples * np.dot( self.dZ, np.transpose( self.A_prev ) )
        self.db = 1 / self.nsamples * np.sum( self.dZ, axis=1, keepdims=True )
        self.dA_prev = np.dot( np.transpose( self.W ), self.dZ )
        return self.dA_prev
        
    def Update( self, alpha ):
        self.W = self.W - alpha * self.dW
        self.b = self.b - alpha * self.db

# could this be derived from FeedFwdLayer?
class InputLayer:
        
    def __init__( self, data ):
        self.Z = 0
        self.dZ = 0
        self.dW = 0
        self.W = 0
        self.b = 0
        self.A = data
        self.nnodes = np.shape( self.A )[0]
    
    def SetData( self, data ):
        self.A = data
        self.nnodes = np.shape( self.A )[0]
        
    def GetNNodes( self ):
        return self.nnodes
        
    def ForwardProp( self, X ):
        return self.A
        
    def BackwardProp( self, dA ):
        return None
        
    def Update( self, alpha ):
        pass
        
