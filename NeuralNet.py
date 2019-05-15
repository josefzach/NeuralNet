import LossFunctions
import numpy as np
import matplotlib.pyplot as plt

class NeuralNet:
    def __init__( self, LearningRate=0.5, InitMethod='He', RegularizationParam=0.0, KeepProb=1.0 ):
        self.nnLayers = list()
        self.nLayers = 0

        # Hyperparameters
        self.initMethod = InitMethod
        self.learningRate = LearningRate
        self.regularizationParam = RegularizationParam
        self.keepProb = KeepProb

        self.Cost = np.array([[]])
        
    def Append( self, nnLayer ):
        self.nnLayers.append( nnLayer )
        
        if self.nLayers > 0:      
            self.nnLayers[-1].SetNNodesPrev( self.nnLayers[-2].GetNNodes() )
            self.nnLayers[-1].Initialize( learningRate=self.learningRate, initMethod=self.initMethod, keepProb=self.keepProb )
    
        self.nLayers += 1

    def ForwardProp( self ):
        A = None
        for i in range( self.nLayers ):
            A = self.nnLayers[i].ForwardProp( A )

    def BackwardProp( self ):
        dA = None
        for i in range( self.nLayers-1, -1, -1 ):
            dA = self.nnLayers[i].BackwardProp( dA )  

    def UpdateParams( self ):
        for i in range( self.nLayers ):
            #print( i )
            A = self.nnLayers[i].UpdateParams()
        
    def ComputeCost( self ):
        loss = self.nnLayers[-1].L
        m = np.shape( loss )[1]
        C = -1/m * np.sum( loss )
        FN = 0
        for i in range( 1, self.nLayers-1 ):
            FN = FN + np.linalg.norm( self.nnLayers[i].W )
        C = C + self.regularizationParam / (2*m) * FN
        self.Cost = np.append( self.Cost, np.array([[C]]) )

    def Train( self, nEpochs ):
        for i in range( 0, nEpochs+1 ):
            self.ForwardProp()
            self.BackwardProp()
            self.UpdateParams()
            self.ComputeCost()

            if i % 100 == 0:
                print( '=> Epoch %i: Cost=%.6f, w1=%s' % (i,self.Cost[-1],np.array2string(self.nnLayers[1].W.T)) )       
                #print( 'BackwardProp: dW=%s, db=%s' % (np.array2string(self.nnLayers[1].dW.T),np.array2string(self.nnLayers[1].db.T)) )        
   
    def Eval( self, X, Y ):
        self.nnLayers[0].SetData( X )
        self.nnLayers[-1].SetData( Y )
        self.ForwardProp()
        Yest = ( self.nnLayers[-1].A >= 0.5 )
        rateCorrect = np.mean( Yest == Y )
        return rateCorrect
    
    def Reset( self ):
        pass

    def Info( self ):
        print( '============== NetInfo ================')
        print( 'Learning Rate: %f' % (self.learningRate) )
        print( 'Initialization Method: %s' % (self.initMethod) )
        print( 'Keep Probability: %f' % (self.keepProb) )
        
    def Draw( self ):
        fig, ax = plt.subplots() # note we must use plt.subplots, not plt.subplot
        for i in range( self.nLayers-1 ):
            for j in range( self.nnLayers[i].nnodes ):
                xpos = 0+i*1
                ypos = 0-(self.nnLayers[i].nnodes-1)/2+j
                ax.plot(xpos, ypos, 'o',fillstyle='none',markersize=40, markeredgewidth=0.5, markeredgecolor=(0,0,0,0.5))


        #circle2 = plt.Circle((0.5, 0.5), 0.2, color='blue')
        #circle3 = plt.Circle((1, 1), 0.2, color='g', clip_on=False)


        # (or if you have an existing figure)
        # fig = plt.gcf()
        # ax = fig.gca()

        #ax.add_artist(circle1)
        #ax.add_artist(circle2)
        #ax.add_artist(circle3)
        plt.ylim(-1, 1)
        plt.xlim(-1, self.nLayers-1 )
        plt.show()
