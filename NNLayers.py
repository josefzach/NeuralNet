import numpy as np
import ActivationFunctions
import LossFunctions

class NNLayer:
    def __init__( self ):
        self.W = None
        self.dW = None
        self.b = None
        self.db = None
        self.Z = None
        self.dZ = None
        self.A = None
        self.dA = None
        self.nnodes = None
        
    def Initialize( self ):
        #raise NotImplementedError( 'Class %s does not implement method GetNNodes()' % (self.__class__.__name__) )
        pass
    def ForwardProp( self, A_prev ):
        #raise NotImplementedError( 'Class %s does not implement method ForwardProp()' % (self.__class__.__name__) )
        pass
    def BackwardProp( self, dA ):
        #raise NotImplementedError( 'Class %s does not implement method BackwardProp()' % (self.__class__.__name__) )
        pass
    def UpdateParams( self, alpha ):
        #raise NotImplementedError( 'Class %s does not implement method UpdateParams()' % (self.__class__.__name__) )   
        pass
    def GetNNodes( self ):
        #raise NotImplementedError( 'Class %s does not implement method GetNNodes()' % (self.__class__.__name__) )
        pass

class FeedFwdLayer( NNLayer ):
    def __init__( self, nnodes, actfcn ):
        super( FeedFwdLayer, self ).__init__()
        self.nnodes_prev = None
        self.A_prev = None
        self.dA_prev = None
        self.nsamples = None
        
        self.nnodes = nnodes
        self.actfcn = ActivationFunctions.Factory( actfcn )
        
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
    
    def Initialize( self ):
        self.W = np.random.randn( self.nnodes,self.nnodes_prev ) * 0.01
        self.b = np.zeros( ( self.nnodes, 1 ) )
        
    def ForwardProp( self, A_prev ):
        self.nsamples = np.shape( A_prev )[1]    # TODO: do not compute nsamples on each epoch
        self.A_prev = A_prev
        self.Z = np.dot( self.W, self.A_prev ) + self.b
        self.A = self.actfcn( self.Z )
        return self.A
        
    def BackwardProp( self, dA ):
        self.dA = dA
        self.dZ = np.multiply( self.dA, self.actfcn.gradient( self.Z ) )
        self.dW = 1 / self.nsamples * np.dot( self.dZ, np.transpose( self.A_prev ) )
        self.db = 1 / self.nsamples * np.sum( self.dZ, axis=1, keepdims=True )
        self.dA_prev = np.dot( np.transpose( self.W ), self.dZ )
        return self.dA_prev
        
    def UpdateParams( self, alpha ):
        self.W = self.W - alpha * self.dW
        self.b = self.b - alpha * self.db

class InputLayer( NNLayer ):
    def __init__( self, X ):
        super( InputLayer, self ).__init__ ()
        self.X = X
        self.nnodes = np.shape( self.X )[0]

    def SetData( self, X ): # requires update of A_prev in following layer!!
        self.X = X
        self.nnodes = np.shape( self.X )[0]
        
    def GetNNodes( self ):
        return self.nnodes
        
    def ForwardProp( self, A_prev ):
        return self.X
        
    def BackwardProp( self, dA ):
        return None
        
class OutputLayer( NNLayer ):
    def __init__( self, Y, lossFcn ):
        super( OutputLayer, self ).__init__ ()
        
        self.Y = Y
        self.lossFcn = LossFunctions.Factory( lossFcn )
        self.L = None
        self.nnodes_prev = None
    
    def ForwardProp( self, A_prev ):
        self.nsamples = np.shape( A_prev )[1]
        self.A_prev = A_prev
        self.L = self.lossFcn( self.Y, self.A_prev )
        return self.L
        
    def BackwardProp( self, dA ):
        return -self.lossFcn.gradient( self.Y, self.A_prev )  #why negative gradient?

    def SetNNodesPrev( self, nnodes_prev ):
        self.nnodes_prev = nnodes_prev

