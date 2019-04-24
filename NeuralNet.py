import CostFunctions

class NeuralNet:
    def __init__( self ):
        self.nnLayers = list()
        self.numLayers = 0
        
        self.Cost = numpy.array([])
        

    def Append( self, nnLayer ):
        self.nnLayers.append( nnLayer )
               
        self.nnLayers[-1].SetNNodesPrev( self.nnLayers[-2].GetNNodes() )
        self.nnLayers[-1].InitParams()
            
        self.numLayers += 1

    def ForwardProp( self ):
        A = None
        for i in range( self.numLayers ):
            A = self.nnLayers[i].ForwardProp( A )

    def BackwardProp( self, dA ):
        dA = None
        for i in range( self.numLayers-1, -1, -1 ):
            dA = self.nnLayers[i].BackwardProp( dA )  

    def UpdateParams( self, alpha )
        for i in range( self.numLayers ):
            #print( i )
            A = self.nnLayers[i].UpdateParams( alpha )
        
    def ComputCost( self ):
        C = -1/np.shape( self.nnLayers[-1].A )[1] * np.sum( CrossEntropyFcn( Y, A ) )

    def Iterate( self, nEpochs ):
        for i in range( 0, nEpochs ):
            self.ForwardProp()
            self.BackwardProp()
            self.UpdateParams()
            self.ComputeCost()
            
    def Reset( self ):
        pass