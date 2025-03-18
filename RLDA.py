#%%
import pylab as pl
import numpy as np
import scipy as sp
from numpy.linalg import eig
from scipy.io import loadmat
import pdb
#%%
def load_data(fname):
    # load the data
    data = loadmat(fname)
    # extract images and labels
    X = data['X']
    Y = data['Y']
    # collapse the time-electrode dimensions
    X = np.reshape(X,(X.shape[0]*X.shape[1],X.shape[2]))
    # transform the labels to (-1,1)
    Y = np.sign((Y[0,:]>0) -.5)
    # pick only first 500 (1000, 3000) datapoints and compare optimal shrinkage
    X = X[:, :500]
    Y = Y[:500]
    print(X.shape)
    return X,Y
#%%
def crossvalidate_nested(X,Y,f,gammas):
    ''' 
    Optimize shrinkage parameter for generalization performance 
    Input:	X	data (dims-by-samples)
                Y	labels (1-by-samples)
                f	number of cross-validation folds
                gammas	a selection of shrinkage parameters
                trainfunction 	trains linear classifier, returns weight vector and bias term
    '''
    # the next two lines reshape vector of indices in to a matrix:
    # number of rows = # of folds
    # number of columns = # of total data-points / # folds
    N = f*int(np.floor(X.shape[-1]/f))
    idx = np.reshape(np.arange(N),(f,int(np.floor(N/f))))
    pdb.set_trace()
    acc_test = np.zeros((f))
    testgamma = np.zeros((gammas.shape[-1],f))
    
    # loop over folds:
    # select one row of 'idx' for testing, all other rows for training
    # call variables (indices) for training and testing 'train' and 'test'
    for ifold in np.arange(f):
        test=idx[ifold, :]
        train=idx[np.arange(f) != ifold, :].flatten()
        
        # loop over gammas
        for igamma in range(gammas.shape[-1]):
            # each gamma is fed into the inner CV via the function 'crossvalidate_lda'
            # the resulting variable is called 'testgamma'
            testgamma[igamma,ifold] =crossvalidate_lda(X[:,train],Y[train],f-1,gammas[igamma])
        # find the the highest accuracy of gammas for a given fold and use it to train an LDA on the training data
        hgamma_idx=testgamma[:,ifold].argmax()
        hgamma =gammas[hgamma_idx]
        w,b = train_lda(X[:,train],Y[train],hgamma)
        # calculate the accuracy for this LDA classifier on the test data
        pred=np.sign(w.dot(X[:,test]) -b)
        acc_test[ifold] = (pred ==Y[test]).mean()

    # do some plotting
    pl.figure()
    pl.boxplot(testgamma.T)
    pl.xticks(np.arange(gammas.shape[-1])+1,gammas)
    pl.xlabel('$\gamma$')
    pl.ylabel('Accuracy')
    pl.savefig('cv_nested-boxplot.pdf')
    pl.show()
    return acc_test,testgamma
#%%
def crossvalidate_lda(X,Y,f,gamma):
    ''' 
    Test generalization performance of shrinkage lda
    Input:	X	data (dims-by-samples)
                Y	labels (1-by-samples)
                f	number of cross-validation folds
                trainfunction 	trains linear classifier, returns weight vector and bias term
    '''
    N = f*int(np.floor(X.shape[-1]/f))
    idx = np.reshape(np.arange(N),(f,int(np.floor(N/f))))
    acc_test = np.zeros((f))
    
    # loop over folds
    # select one row of idx for testing, all others for training
    # call variables (indices) for training and testing 'train' and 'test'
    for ifold in np.arange(f):
        test=idx[ifold,:]
        train=idx[np.arange(f)!=ifold, :]
        # train LDA classifier with training data and given gamma:
        w, b = train_lda(X[:, train],Y[train],gamma)
        # test classifier on test data:
        pred=np.sign(w.dot(X[:, test])-b)
        acc_test[ifold] = (pred==Y[test]).mean()
    return acc_test.mean()
#%%
def train_lda(X,Y,gamma):
    '''
    Train a nearest centroid classifier
    '''
    # class means
    mupos = np.mean(X[:,Y>0],axis=1)
    muneg = np.mean(X[:,Y<0],axis=1)

    # inter and intra class covariance matrices
    Sinter = np.outer(mupos-muneg,mupos-muneg)
    #Sinter = sp.outer(muneg-mupos,muneg-mupos)
    Sintra = np.cov(X[:,Y>0]) + np.cov(X[:,Y<0])
    # shrink covariance matrix estimate
    Sintra = (1 - gamma) * Sintra + gamma * (np.trace(Sintra) / Sintra.shape[0]) * np.eye(Sintra.shape[0])
    # solve eigenproblem
    eigvals, eigvecs = sp.linalg.eig(Sinter,Sintra)
    # weight vector
    w = eigvecs[:,eigvals.argmax()]
    if np. dot (w, mupos)<np.dot (w, muneg) :
        w =-w
    # offset
    b = (w.dot(mupos) + w.dot(muneg))/2.
    # return the weight vector
    return w,b
#%%
X,Y = load_data('bcidata.mat')
gammas=np.array([0,.005,.05,.5,1])
a,b = crossvalidate_nested(X,Y,10,gammas)
print(a)
print(b)
