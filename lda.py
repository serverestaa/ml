#%%
import pylab as pl
import scipy as sp
import numpy as np
from scipy.linalg import eig
from scipy.io import loadmat
import pdb
#%%
def load_data(fname):
    # load the data
    data = loadmat(fname)
    X,Y = data['X'],data['Y']
    # collapse the time-electrode dimensions
    X = np.reshape(X,(X.shape[0]*X.shape[1],X.shape[2]))
    # transform the labels to (-1,1)
    Y = np.sign((Y[0,:]>0) -.5)
    return X,Y
#%%
X,Y = load_data('bcidata.mat')
print(X.shape)
print(Y.shape)

#%%
def train_ncc(X,Y):
    '''
    Train a nearest centroid classifier
    '''
    muone= X[:, Y>0].mean(axis=1)
    mutwo=X[:,Y < 0].mean(axis=1)
    w= muone -mutwo
    b =(muone.T@muone -mutwo.T@ mutwo)/2
    return w,b
#%%
def train_lda(X,Y):
    '''
    Train a linear discriminant analysis classifier
    '''
    muone =X[:,Y> 0].mean(axis=1)
    mutwo= X[:,Y <0].mean(axis=1)
    S_B=np.outer(muone - mutwo, muone - mutwo)
    S_W =np.zeros((X.shape[0], X.shape[0]))
    N_one=np.sum(Y>0)
    N_two=np.sum(Y<0)

    for xone in X[:,Y > 0].T:
        S_W+=np.outer(xone - muone, xone - muone)/N_one

    for xtwo in X[:,Y < 0].T:
        S_W+=np.outer(xtwo - mutwo, xtwo - mutwo)/N_two

    eigvals,eigvecs =eig(S_B,S_W)
    w = eigvecs[:,np.argmax(eigvals)].real
    if np.dot(w, muone)<np.dot(w, mutwo):
        #I added this lines because I got interesting anomaly:
        # my eigenvectors are sometimes oriented opposite of target class and even if
        # I had good separation my accuracy was 2-3%
        w = -w

    b=w.T@(muone+mutwo) /2

    return w,b
#%%
def compare_classifiers():
    '''
    compares nearest centroid classifier and linear discriminant analysis
    '''
    fname = 'bcidata.mat'
    X,Y = load_data(fname)

    permidx = np.random.permutation(np.arange(X.shape[-1]))
    trainpercent = 70.
    stopat = int(np.floor(Y.shape[-1]*trainpercent/100.))
    #pdb.set_trace()
    
    X,Y,Xtest,Ytest = X[:,permidx[:stopat]],Y[permidx[:stopat]],X[:,permidx[stopat:]],Y[permidx[stopat:]]

    w_ncc,b_ncc = train_ncc(X,Y)
    w_lda,b_lda = train_lda(X,Y)
    fig = pl.figure(figsize=(12,5))

    ax1 = fig.add_subplot(1,2,1)
    #pl.hold(True)
    ax1.hist(w_ncc.dot(Xtest[:,Ytest<0]))
    ax1.hist(w_ncc.dot(Xtest[:,Ytest>0]))
    ax1.set_xlabel('$w^{T}_{NCC}X$')
    ax1.legend(('non-target','target'))
    ax1.set_title("NCC Acc " + str(np.sum(np.sign(w_ncc.dot(Xtest)-b_ncc)==Ytest)*100/Xtest.shape[-1]) + "%")
    ax2 = fig.add_subplot(1,2,2)
    ax2.hist(w_lda.dot(Xtest[:,Ytest<0]))
    ax2.hist(w_lda.dot(Xtest[:,Ytest>0]))
    ax2.set_xlabel('$w^{T}_{LDA}X$')
    ax2.legend(('non-target','target'))
    ax2.set_title("LDA Acc " + str(np.sum(np.sign(w_lda.dot(Xtest)-b_lda)==Ytest)*100/Xtest.shape[-1]) + "%")
    pl.savefig('ncc-lda-comparison.pdf')
    pl.show()

#%%
compare_classifiers()
#%%
def crossvalidate(X,Y,f=10,trainfunction=train_lda):
    ''' 
    Test generalization performance of a linear classifier
    Input:	X	data (dims-by-samples)
            Y	labels (1-by-samples)
            f	number of cross-validation folds
            trainfunction 	trains linear classifier
    '''
    N=X.shape[1]
    idx= np.random.permutation(N)
    fold_size =N//f
    acc_train=np.zeros(f)
    acc_test = np.zeros(f)

    for ifold in range(f):
        test=idx[ifold*fold_size:(ifold+1)*fold_size]
        train =np.setdiff1d(idx, test)

        Xtrain,Ytrain = X[:,train],Y[train]
        Xtest, Ytest=X[:,test],Y[test]
        # train classifier
        w,b =trainfunction(X[:,train],Y[train])
        # compute accuracy on training data
        acc_train[ifold]=np.mean(np.sign(np.dot(w,Xtrain) - b)==Ytrain) * 100
        # compute accuracy on test data
        acc_test[ifold]=np.mean(np.sign(np.dot(w,Xtest) - b)==Ytest) * 100

    return acc_train,acc_test


#%%
X,Y = load_data('bcidata.mat')
acc_train,acc_test=crossvalidate(X,Y,f=10,trainfunction=train_lda)
print("===========================")
print("train")
print(acc_train)
print("===========================")
print("test")
print(acc_test)
print("===========================")
data = [acc_train, acc_test]
pl.figure(figsize=(6,6))
pl.boxplot(data)
pl.xticks([1, 2], ["train block", "test block"])
pl.ylabel("Accuracy")
pl.title("Train vs. Test Accuracy")
pl.show()