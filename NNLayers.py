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

        self.learningRate = None
        
    def Initialize( self ):
        #raise NotImplementedError( 'Class %s does not implement method GetNNodes()' % (self.__class__.__name__) )
        pass
    def ForwardProp( self, A_prev ):
        #raise NotImplementedError( 'Class %s does not implement method ForwardProp()' % (self.__class__.__name__) )
        pass
    def BackwardProp( self, dA ):
        #raise NotImplementedError( 'Class %s does not implement method BackwardProp()' % (self.__class__.__name__) )
        pass
    def UpdateParams( self ):
        #raise NotImplementedError( 'Class %s does not implement method UpdateParams()' % (self.__class__.__name__) )   
        pass
    def GetNNodes( self ):
        #raise NotImplementedError( 'Class %s does not implement method GetNNodes()' % (self.__class__.__name__) )
        pass

class FeedFwdLayer( NNLayer ):
    def __init__( self, nnodes, actFunction='Sigmoid', dropRate=0.0 ):
        super( FeedFwdLayer, self ).__init__()
        self.nnodes_prev = None
        self.A_prev = None
        self.dA_prev = None
        self.nsamples = None
        self.D = None
        self.drop_rate = dropRate
        
        self.nnodes = nnodes
        self.actfcn = ActivationFunctions.Factory( actFunction )
        
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
    
    def Initialize( self, learningRate=0.5, initMethod='He', keepProb=1.0 ):
        if initMethod == 'He':
            self.W = np.random.randn( self.nnodes,self.nnodes_prev ) * np.sqrt( 2 / self.nnodes_prev )
        if initMethod == 'Xavier':
            self.W = np.random.randn( self.nnodes,self.nnodes_prev ) * np.sqrt( 1 / self.nnodes_prev )
        else:
            self.W = np.random.randn( self.nnodes,self.nnodes_prev ) * 0.01
        
        
        self.b = np.zeros( ( self.nnodes, 1 ) )
        print( 'Initializing Layer n with: W=%s, b= %s' % (np.array2string(self.W.T), np.array2string(self.b.T)) )

        self.learningRate = learningRate
        self.keepProb = keepProb
        
    def ForwardProp( self, A_prev ):
        self.nsamples = np.shape( A_prev )[1]    # TODO: do not compute nsamples on each epoch
        self.A_prev = A_prev
        self.Z = np.dot( self.W, self.A_prev ) + self.b
        self.A = self.actfcn( self.Z )
        self.D = np.random.rand( self.A.shape[0], self.A.shape[1] )
        self.D = self.D > self.drop_rate
        self.A = self.A * self.D / (1-self.drop_rate)
        return self.A
        
    def BackwardProp( self, dA ):
        self.dA = dA * self.D / (1-self.drop_rate)
        self.dZ = np.multiply( self.dA, self.actfcn.gradient( self.Z ) )
        self.dW = 1 / self.nsamples * np.dot( self.dZ, np.transpose( self.A_prev ) )
        self.db = 1 / self.nsamples * np.sum( self.dZ, axis=1, keepdims=True )
        self.dA_prev = np.dot( np.transpose( self.W ), self.dZ )      
        return self.dA_prev
        
    def UpdateParams( self ):
        lambd = 0
        self.W = self.W - self.learningRate * ( self.dW + lambd/self.nnodes*self.W )
        self.b = self.b - self.learningRate * self.db

class InputLayer( NNLayer ):
    def __init__( self, X ):
        super( InputLayer, self ).__init__ ()
        self.X = X
        self.nnodes = np.shape( self.X )[0]

    def SetData( self, X ):
        self.X = X
        self.nnodes = np.shape( self.X )[0]
        # catch case where shape(x)[0] changes --> re-training network required
        
    def GetNNodes( self ):
        return self.nnodes
        
    def ForwardProp( self, A_prev ):
        return self.X
        
    def BackwardProp( self, dA ):
        return None

class OutputLayer( FeedFwdLayer ): #rather inherit from non-dropout layer?
    def __init__( self, Y, nnodes, actFunction='Sigmoid', lossFcn='CrossEntropy' ):
        super( OutputLayer, self ).__init__( nnodes, actFunction )

        self.Y = Y 
        self.lossFcn = LossFunctions.Factory( lossFcn )
        self.L = None

    def ForwardProp( self, A_prev ):
        self.nsamples = np.shape( A_prev )[1]  # TODO: do not compute nsamples on each epoch
        self.A_prev = A_prev
        self.Z = np.dot( self.W, self.A_prev ) + self.b

        self.A = self.actfcn( self.Z )
        
        self.L = self.lossFcn( self.Y, self.A )

        return self.A
        
    def BackwardProp( self, dA ):
        self.dA = -self.lossFcn.gradient( self.Y, self.A )  #why negative gradient?
        self.dZ = np.multiply( self.dA, self.actfcn.gradient( self.Z ) )
        self.dW = 1 / self.nsamples * np.dot( self.dZ, np.transpose( self.A_prev ) )
        self.db = 1 / self.nsamples * np.sum( self.dZ, axis=1, keepdims=True )
        self.dA_prev = np.dot( np.transpose( self.W ), self.dZ )      
        return self.dA_prev

    def SetData( self, Y ):
        self.Y = Y
        # catch dimension missmatch
