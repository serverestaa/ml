#%%
import matplotlib.pylab as pl
import scipy as sp
import numpy as np
from scipy.io import loadmat
import pdb
#%%
def load_data(fname):
    # load the data
    data = loadmat(fname)
    # extract images and labels
    imgs = data['data_patterns']
    labels = data['data_labels']
    return imgs, labels
#%%
def ncc_train(X,Y,Xtest,Ytest):
    # initialize accuracy vector
    acc = np.zeros(X.shape[-1])
    # unique class labels
    cids = np.unique(Y)
    # initialize mu, shape should be (256,2) - we have 2 centroids and each centroid is a vector with 256 elements
    mu = np.zeros((X.shape[0], len(cids)))
    # initialize counter , shape should be (2,) - we should know what class is this and how many samples we already have that's why we have size 2
    Nk = np.zeros(len(cids))
    # loop over all data points in training set
    for n in np.arange(X.shape[-1]):
        # set idx to current class label
        idx = (cids==Y[n])
        # update mu
        mu[:, idx] = (mu[:, idx] * Nk[idx] + X[:, n].reshape(-1,1)) / (Nk[idx] + 1)
        # update counter
        Nk[idx] = Nk[idx]+1
        # predict test labels with current mu
        yhat=predict_ncc(Xtest,mu)
        # calculate current accuracy with test labels
        acc[n]= np.mean(yhat==Ytest)
    # return weight vector and error
    return mu,acc

#%%
def predict_ncc(X,mu):
    # do nearest-centroid classification
    # initialize distance matrix
    NCdist=np.zeros((X.shape[1], mu.shape[1]))
    # compute euclidean distance to centroids
    # loop over both classes
    for ic in np.arange(mu.shape[-1]):
        # calculate distances of every point to centroid
        #
        NCdist[:, ic]=np.sqrt(np.sum((X - mu[:, ic].reshape(-1, 1))**2, axis=0))
        
    # assign the class label of the nearest (euclidean distance) centroid
    idx = NCdist.argmin(axis=1)
    Yclass = np.where(idx == 0, -1, 1)
    return Yclass

#%%
digit=0

# load the data
fname = "usps.mat"
imgs,labels = load_data(fname)
# we only want to classify one digit 
labels = np.sign((labels[digit,:]>0)-.5)

# please think about what the next lines do
permidx = np.random.permutation(np.arange(imgs.shape[-1]))
trainpercent = 70.
stopat = np.floor(labels.shape[-1]*trainpercent/100.)
stopat= int(stopat)

# cut segment data into train and test set into two non-overlapping sets:
X = imgs[:, permidx[:stopat]]
Y = labels[permidx[:stopat]]
Xtest = imgs[:, permidx[stopat:]]
Ytest = labels[permidx[stopat:]]
#check that chapes of X and Y make sense..

# now comes the model estimation..
mu,acc_ncc = ncc_train(X,Y,Xtest,Ytest)

#%%
#save the results as a figure
fig = pl.figure(figsize=(10,5))
ax1 = fig.add_subplot(1,2,1)
ax1.plot(acc_ncc*100.)
pl.xlabel('Iterations')
pl.title('NCC')
pl.ylabel('Accuracy [%]')

# and imshow the weight vector
ax2 = fig.add_subplot(1,2,2)
# reshape weight vector
weights = np.reshape(mu[:,-1],(int(np.sqrt(imgs.shape[0])),int(np.sqrt(imgs.shape[0]))))
# plot the weight image
imgh = ax2.imshow(weights)
# with colorbar
pl.colorbar(imgh)
ax2.set_title('NCC Centroid')
# remove axis ticks
pl.xticks(())
pl.yticks(())
# remove axis ticks
pl.xticks(())
pl.yticks(())

# write the picture to pdf
fname = 'NCC_digits-%d.pdf'%digit
pl.savefig(fname)
#%%
