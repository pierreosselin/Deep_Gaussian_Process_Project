import numpy as np
import deepgp
import matplotlib.pyplot as plt

class BayesianOptimization():
    """
    Class for performing bayesian optimization with DGPs

    Initialization
    ---------
    model : DeepGP arguments [nDims, kernels, num_inducing]
    f : Function expensive to evaluate
    Xini (n x (d+1)) : Initial points (first dimension is the observation), if not none, the model is supposed to be untrained, if None, the model is supposed to be pretrained.
    nSamples : Number of Samples for the EI estimation
    max_iters : Max number of iterations for the optimization process
    xbounds : [1 x d np.array, 1 x d np.array]

    """
    def __init__(self, model, f,yini, Xini, nSamples, max_iters, xbounds):
        self.model = model
        self.f = f
        self.nSamples = nSamples
        self.max_iters = max_iters
        self.xmin = xbounds[0]
        self.xmax = xbounds[1]
        self.x = Xini
        self.y = yini
        self.ymin = yini.min()
        self.xbounds = xbounds
        ## To do : treat the case where there is no initialization uniform?
        self._update()

    def _update(self):
        self.offset = self.y.mean()
        self.scale = np.sqrt(self.y.var())
        self.yhat = (self.y-self.offset)/self.scale
        print("Train model with")
        print("-" + str(self.model[2]) + " Inducing points")
        print("-" + str(self.x.shape[0]) + "Observation points")
        print(self.x)
        print(self.y)
        self.m = deepgp.DeepGP(self.model[0], Y=self.yhat, X=self.x, kernels=self.model[1], num_inducing=self.model[2], back_constraint=False)
        for i in range(len(self.m.layers)):
            output_var = self.m.layers[i].Y.var() if i==0 else self.m.layers[i].Y.mean.var()
            self.m.layers[i].Gaussian_noise.variance = output_var*0.001
            self.m.layers[i].Gaussian_noise.variance.fix()
        self.m.optimize(max_iters=self.max_iters)

    def compute_InfillEI(self, Xtest): #Xtest must be by column
        """
        Compute the infill Expectation improvement at the X test points with a given number of samples_prediction
        Return : 1d Array of the estimated EI
        """
        samples = (self.m.samples_prediction(Xtest, nSamples = self.nSamples)).reshape(self.nSamples, Xtest.shape[0])
        samples = self.ymin - (samples*self.scale**2 + self.offset)
        samples = samples*(samples > 0.)
        ei = samples.mean(axis = 0)
        self.currentEI = [Xtest, ei]
        self.flag = True
        return ei

    def compute_predictive(self,Xtest):
        return self.m.predict(Xtest)

    def compute_nextpoint(self):
        if self.flag:
            index = np.argmax(self.currentEI[1])
            self.nextpoint = self.currentEI[0][index]
            return self.currentEI[0][index]
        else:
            print("Compute the EI first !")

    def update_model(self):
        self.model[2] += 1 #Increase inducing points
        evaluation = self.f(self.nextpoint)
        self.flag = False
        self.y = np.vstack((self.y, evaluation))
        self.x = np.vstack((self.x, self.nextpoint))
        self.ymin = self.y.min()
        self._update()

    def perform_step(self):
        Xtest = np.linspace(0,1,500)
        self.compute_InfillEI(Xtest[:,None])
        self.compute_nextpoint()
        self.plot()
        self.update_model()

    def plot(self):
        Xtest = np.linspace(self.xbounds[0][0], self.xbounds[1][0],1000)
        ## Fix this for higher dimension
        Ypredict = self.m.predict(Xtest[:,None])
        fig, axs = plt.subplots(2, figsize = (10,15))
        axs[0].plot(Xtest, Ypredict[0]*self.scale + self.offset)
        axs[0].fill_between(Xtest, Ypredict[0][:,0]*self.scale + self.offset-2*Ypredict[1][:,0]*self.scale**2, Ypredict[0][:,0]*self.scale + self.offset+2*Ypredict[1][:,0]*self.scale**2, color='red', alpha=0.15, label='$2 \sigma_{2|1}$')
        axs[0].plot(self.x, self.y, '.')
        axs[0].axvline(x=self.nextpoint,color = 'r')
        axs[1].plot(self.currentEI[0], self.currentEI[1])
        axs[1].axvline(x=self.nextpoint,color = 'r')
