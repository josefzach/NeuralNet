class NeuralNet:
    def __init__( self ):
        self.nnLayers = list()
        self.numLayers = 0
        pass

    def Append( self, nnLayer ):
        self.nnLayers.append( nnLayer )
               
        self.nnLayers[-1].SetNNodesPrev( self.nnLayers[-2].GetNNodes() )
        self.nnLayers[-1].InitParams()
            
        self.numLayers += 1

    def ForwardProp( self ):
        A = None
        for i in range( nlayers ):
            A = DNN[i].ForwardProp( A )

    def ComputCost( self ):
        C = -1/np.shape( nnLayers[-1].A )[1] * np.sum( CrossEntropyFcn( Y, A ) )

