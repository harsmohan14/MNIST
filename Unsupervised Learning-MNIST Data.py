
# coding: utf-8

# In[5]:


import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
from scipy.stats import skew
from scipy.stats import kurtosis
import scipy.stats as stat
get_ipython().magic('matplotlib inline')
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_digits
from sklearn.cross_validation import train_test_split


# In[6]:


from scipy.cluster import hierarchy


# In[7]:


digitdata = load_digits()


# In[8]:


digitX = digitdata['data']
digitX.shape


# In[9]:


digitY = digitdata.target


# In[10]:


digitX


# In[11]:


z = hierarchy.linkage(digitX,'complete')
z


# In[12]:


#hierarchy.dendrogram(z,p=10,truncate_mode='lastp',show_contracted=True)
plt.figure(figsize=(10,10))
hierarchy.dendrogram(z)
plt.show()


# In[13]:


"""For calcuating threshold (t), find longest vertical line in dendogram which doesnt cross any horizonal line(assuming 
all horizontal lines extend across the whole of dendograph width. Then draw a horizonal line across that vertical line and the 
intersections would be #of clusters)"""

clusters = hierarchy.fcluster(z,t=10,criterion='maxclust')
len(np.unique(clusters))


# In[14]:


plt.scatter(clusters,digitY)
plt.show()


# In[15]:


from sklearn.preprocessing import StandardScaler


# In[16]:


#Doing scaling
scalar = StandardScaler()


# In[17]:


digit_sc_X = scalar.fit_transform(digitX)
linkage = hierarchy.linkage(digit_sc_X,'complete')
clusters1 = hierarchy.fcluster(linkage,criterion='maxclust',t=10)
plt.scatter(clusters1, digitY)
plt.show()
#Not able to understand a large clusters, so it is not efficient


# In[18]:


len(digit_sc_X)


# In[19]:


digit_sc_X.shape


# In[20]:


#Trying with sklearn KMeans

from scipy.cluster import vq
from sklearn.cluster import KMeans
from sklearn import metrics


# In[21]:


skmeans = KMeans(10)


# In[22]:


skmeans.fit(digit_sc_X)
#n_init denotes the number of times KMeans algo will run with different random centroids in starting


# In[23]:


np.unique(skmeans.labels_)


# In[24]:


#Use silhouette_score to find accuracy. Elbow method is another technique to find optimal num of clusters
z=0
score = [0]*len(range(5,20))
for i in range(5,20):
    kmean = KMeans(i)
    kmean.fit(digit_sc_X)
    score[z] = metrics.silhouette_score(digit_sc_X,kmean.labels_)
    z+=1


# In[25]:


score


# In[26]:


plt.plot(range(5,20),score)
plt.show()


# In[27]:


#Calculating wcss (Within Cluster Sum of Squares) using KMeans and elbow method
#wcss = [0]*len(range(5,20))
wcss = []
for i in range(1,20):
    kmean = KMeans(i)
    kmean.fit(digit_sc_X)
    wcss.append(kmean.inertia_)


# In[28]:


plt.plot(range(1,20),wcss)
plt.show()


# In[29]:


#Calculating wcss without scaling on train data
#wcss = [0]*len(range(5,20))
wcss1 = []
for i in range(1,20):
    kmean = KMeans(i,random_state=10)
    kmean.fit(digitX)
    wcss1.append(kmean.inertia_)


# In[30]:


plt.plot(range(1,20),wcss1)
plt.show()


# In[31]:


digitX.shape


# In[32]:


kmean = KMeans(n_clusters=10,random_state=10)
predicted = kmean.fit_predict(digitX)
predicted


# In[33]:


kmean.cluster_centers_[2]


# In[34]:


#It shows without logistic regression, it fails to predict
print (metrics.classification_report(digitY,predicted))


# In[35]:


print (metrics.confusion_matrix(digitY,predicted))


# In[36]:


#Using MDS to understand clustering and correlation among observations
from sklearn.manifold import MDS
from sklearn.metrics.pairwise import euclidean_distances


# In[37]:


digit_arr = digit_sc_X.tolist()
dist = euclidean_distances(digit_arr,digit_arr)


# In[38]:


mds = MDS(dissimilarity='precomputed',random_state=30)
mdsX = mds.fit_transform(dist)


# In[39]:


mdsarrX = np.array(mdsX)
plt.scatter(mdsarrX[:,0],mdsarrX[:,1],c=digitY)
plt.xlim([-15,15])
plt.ylim([-20,20])
plt.show()
#Too much overlapping in graph, so its not suitable for modeling yet


# In[40]:


len(mdsarrX[:,:0])


# In[41]:


len(mdsarrX[:,0])


# In[44]:


#Using LDA(Linear Discriminant Analysis)
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA


# In[45]:


lda = LDA(n_components=2)


# In[47]:


X_train = lda.fit_transform(digit_sc_X, digitY)


# In[49]:


Y_train = lda.transform(digitY.reshape(-1,1))
#Now put X_train and Y_train in model for training


# In[90]:


#Using sklearn PCA
from sklearn.decomposition import PCA


# In[91]:


pca1 = PCA(0.99)


# In[92]:


digit_sc_X.shape


# In[93]:


pca1.fit(digit_sc_X)
pcomponents = pca1.fit_transform(digit_sc_X)
pca1.n_components_


# In[94]:


pca1.explained_variance_ratio_


# In[95]:


pca1.components_.shape


# In[96]:


pcomponents.shape


# In[97]:


#Using PCA and Logistic regression to build model


# In[98]:


digittrainX, digittestX,digittrainY,digittestY = train_test_split(digitdata['data'],digitdata.target, test_size=0.2,random_state=43)


# In[99]:


digittrainX.shape


# In[100]:


digittestX.shape


# In[101]:


pca2 = PCA(0.99)


# In[102]:


scalar1 = StandardScaler()
digittrain_sc_X = scalar1.fit_transform(digittrainX)


# In[103]:


#No scalar fit to be done on test data
digittest_sc_X = scalar1.transform(digittestX)


# In[104]:


digittrain_pca_X = pca2.fit_transform(digittrain_sc_X)


# In[105]:


#No PCA fit to be done on test data
digittest_pca_X = pca2.transform(digittest_sc_X)


# In[106]:


digittrain_pca_X.shape


# In[107]:


digittest_pca_X.shape


# In[108]:


from sklearn.linear_model import LogisticRegression


# In[109]:


logreg = LogisticRegression()


# In[110]:


logmodel2 = logreg.fit(digittrain_pca_X,digittrainY)


# In[111]:


predicted2 = logmodel2.predict(digittest_pca_X)
predicted2


# In[112]:


from sklearn import metrics


# In[113]:


print (metrics.classification_report(digittestY,predicted2))


# In[114]:


kmeans = KMeans(10)


# In[115]:


digitKmeans = kmeans.fit(digittrain_pca_X)


# In[116]:


plt.scatter(digitKmeans.labels_, digittrainY)
plt.show()


# In[119]:


predicted4 = digitKmeans.predict(digittrain_pca_X)
print (metrics.classification_report(digittrainY, predicted4))


# In[173]:


digittrain_pca_X_list = digittrain_pca_X.tolist()
digitdist1 = euclidean_distances(digittrain_pca_X_list,digittrain_pca_X_list)


# In[174]:


mds1 = MDS(n_components=2,dissimilarity='precomputed',random_state=30)


# In[175]:


mds1Fit = mds1.fit_transform(digitdist1)


# In[176]:


mds1arr = np.array(mds1Fit)
mds1arr


# In[222]:


plt.scatter(mds1arr[:,0],mds1arr[:,1],c=digittrainY)
plt.xlim([-15,15])
plt.ylim([-20,20])
plt.show()


# In[146]:


#Modelling without scaling and PCA to check accuracy


# In[147]:


logreg3 = LogisticRegression()


# In[148]:


logmodel3 = logreg3.fit(digittrainX,digittrainY)


# In[149]:


predicted3 = logmodel3.predict(digittestX)


# In[151]:


print (metrics.classification_report(digittestY,predicted3))


# In[178]:


#Calculating PCA without using library


# In[197]:


digittrain_sc1_X = scalar1.fit_transform(digitX)


# In[198]:


digitcov = np.cov(digittrain_sc1_X.T)


# In[199]:


eigenVal, eigenVec = np.linalg.eig(digitcov)
eigenVec.shape


# In[200]:


plt.plot(np.abs(eigenVal))
plt.show()


# In[201]:


eigenVal


# In[202]:


digiteigenVec = eigenVec[:,(eigenVal>0.20)]
digiteigenVec.shape


# In[203]:


digitpca = np.dot(digittrain_sc1_X,digiteigenVec)
digitpca.shape


# In[204]:


kmeans1 = KMeans(10)
kmeans1.fit(digitpca)


# In[205]:


plt.scatter(kmeans1.labels_,digitY)
plt.show()


# In[206]:


#Calculating MDS from manual pca


# In[215]:


digitdist2 = euclidean_distances(digitpca.tolist())


# In[216]:


mds2 = MDS(dissimilarity='precomputed',n_components=2,random_state=20)


# In[217]:


mds2Fit = mds2.fit_transform(digitdist2)


# In[218]:


mds2Fitarr = np.array(mds2Fit)
mds2Fitarr.shape


# In[220]:


plt.scatter(mds2Fitarr[:,0],mds2Fitarr[:,1],c=digitY)
plt.xlim([-15,15])
plt.ylim([-20,20])
plt.show()

