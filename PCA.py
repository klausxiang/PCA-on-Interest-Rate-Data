#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('pylab', 'inline')
import pandas as pd
import numpy as np

import fmt
from numpy import linalg as LA


# In[2]:


cmturl = "https://raw.githubusercontent.com/yadongli/nyumath2048/master/data/cmt.csv"
cmt_rates = pd.read_csv(cmturl, parse_dates=[0], index_col=[0])

cmt_rates.plot(legend=False);
tenors = cmt_rates.columns.map(float)
tenorTags = ['T=%g' % m for m in tenors]


# In[3]:


cmt_rates.head()


# In[4]:


#compute correlation matrix and covariance matrix
daily_change = (cmt_rates - cmt_rates.shift())
daily_change = daily_change[1:]
corr = daily_change.corr()
cov = daily_change.cov()
cov


# In[5]:


# PCA on Cov matrix and Corr matrix
eigval_cov,eigvec_cov = LA.eig(cov)
eigval_corr,eigvec_corr = LA.eig(corr)

print('PC for Cov matrix: ')
pd.DataFrame(eigvec_cov)


# In[6]:


print('PC for Corr matrix: ')
pd.DataFrame(eigvec_corr)


# In[7]:


print('Eigenvalues for Cov matrix:')
print(eigval_cov)
print('Eigenvalues for Corr matrix: ')
print(eigval_corr)


# In[8]:


# In terms of variance explained
varEx_cov = [0.]*len(eigval_cov)
varEx_corr = [0.]*len(eigval_corr)
totalVar_cov = sum(eigval_cov)
totalVar_corr = sum(eigval_corr)

for i in range(len(eigval_cov)):
    varEx_cov[i] = sum(eigval_cov[:i+1])/ totalVar_cov 
    
for i in range(len(eigval_corr)):
    varEx_corr[i] = sum(eigval_corr[:i+1])/ totalVar_corr 


plt.plot(varEx_cov)
plt.plot(varEx_corr)
plt.title('Culmulative vairance explained by PC')
plt.xlabel('n-th PC')
plt.ylabel('Culmulative vairance explained ')
plt.legend(['Cov PCA','Corr PCA'])

    
    


# Normally when the Cov matrix have data on the similar scale, the PCA on Cov and Corr matrix should be equivalent. Here in the above example, we see that the values of eigvectors from 2 PCA are similar, the culmulative variance explained looks similar, so we can say that they are equivalent in the sense of explaining variance

# In[9]:


n_Cov, n_Corr = 0, 0
for i in range(len(varEx_cov)):
    if varEx_cov[i]>=0.95:
        n_Cov = i+1
        break
        
for i in range(len(varEx_cov)):
    if varEx_corr[i]>=0.95:
        n_Corr = i+1
        break

print(n_Cov, ' principal components are required to explain 95% of the variance in Cov matrix')
print(n_Corr, ' principal components are required to explain 95% of the variance in Corr matrix')


        


# In[10]:


# first 3 PC for the Cov matrix
pc1 = eigvec_cov[:,0]
pc2 = eigvec_cov[:,1]
pc3 = eigvec_cov[:,2]

tenors = [0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 20.0]
plt.plot(tenors,pc1)
plt.plot(tenors,pc2)
plt.plot(tenors,pc3)
plt.legend(['1st PC','2nd PC','3rd PC'])
plt.xlabel('Tenors')
plt.title('First 3 PC')
plt.show()


# The first 3 PC have level, steepness, curvature interpretation. 
# 
# The first component representing “level”, the general level of interest rates, where all rates go up or down by the same amount. It shows that the change of yield of long-term bonds with maturity less than 5 yrs is higher than that of shorter-term bonds. 
# 
# The second component representing steepness explains how rapidly the yield curve goes up (or down) from short rates to longer rates. 
# 
# The third component representing “curvature”, which is a butterfly shape that occurs between short, medium, and long-term rates.

# In[11]:


# plot the the factor loading (or the projection) to the first 3 principal components

# Make a list of (eigenvalue, eigenvector) tuples (eigenvectors[:,i] gets i+1 th column)
eigenpairs = [(np.abs(eigval_cov[i]), eigvec_cov[:,i]) for i in range(len(eigval_cov))]

# Sort the (eigenvalue, eigenvector) tuples from high to low
eigenpairs.sort(key=lambda k: k[0], reverse=True)

# construct projection matrix W from the “top” k eigenvectors.
# choose 3 pc b/c they expalin 95% of variace
w = np.hstack((eigenpairs[0][1][:, np.newaxis], eigenpairs[1][1][:, np.newaxis],eigenpairs[2][1][:, np.newaxis]))
print('W=\n', w)


# In[12]:


projection = daily_change.dot(w)
projection


# In[13]:


plt.plot(projection)
plt.legend(['1st PC','2nd PC','3rd PC'])
plt.xlabel('Year')
plt.ylabel('Projected daily change')
plt.title('Projection of historical data to First 3 PC')
plt.show()


# 
# Use the PCA of the covariance matrix: 
# $$
# \scriptsize
# V = R_V \Lambda_V R_V^T \approx \dot R_V \dot \Lambda_V \dot R_V^T = \dot R_V \dot H_V \dot H_V^T  \dot R_V^T = (\overbrace{\dot R_V \dot H_V}^{L_V})(\dot R_V\dot H_V )^T \\
# $$
# 
# 
# Simulation can then be driven by $\delta w = \dot R_V \dot H_V \dot z \sqrt{\delta t}$
# 
# Then we use Vasicek model (O-U process)  $$\delta r = (\mu-r)\delta t + \sigma \delta w_t$$
# where   $\sigma = 1$
# to model our change in rates

# In[14]:


cmt_rates.tail()


# In[15]:


LAMBDA = np.diag([eigval_cov[0],eigval_cov[1],eigval_cov[2]])
R = w
H = np.sqrt(LAMBDA)
L = R@H

dt = 1/250
mu = np.array(np.mean(cmt_rates))
sigma = 1

#  simulate the daily interest rate changes up to the future time of 20Y 
r = np.zeros((250*20,9))
dr = np.zeros((250*20-1,9))

r[0] = np.array(cmt_rates.loc['2014-12-26'])
for i in range(len(r)-1):
    Z = np.random.normal(0,1,3) # 3x3
    dr[i] = (mu-r[i])*dt+sigma*L@Z
    r[i+1] = r[i]+dr[i]


# In[16]:


x = np.array([n/250 for n in range (250*20-1)])
plt.plot(x,dr)
plt.xlabel('Year')
plt.ylabel('dr')
plt.title('Prediction of dr in furture 20 year')
plt.legend(cmt_rates.columns)
plt.show()


# In[17]:


x = np.array([n/250 for n in range (250*20)])
plt.plot(x,r)
plt.xlabel('Year')
plt.ylabel('r')
plt.title('Prediction of r in furture 20 year')
plt.legend(cmt_rates.columns)
plt.show()


# In[18]:


cov_simulated = pd.DataFrame(dr).cov()
cov_simulated.rename(columns={0:'0.25',1:'0.5',2:'1',3:'2',4:'3',5:'5',6:'7',7:'10',8:'20'},
                    index = {0:'0.25',1:'0.5',2:'1',3:'2',4:'3',5:'5',6:'7',7:'10',8:'20'})


# In[19]:


cov = daily_change.cov()
cov


# We can see that the simulated cov matrix using first 3 PC are actually quite similar to our oringal cov matrix. This follows from the fact that thr first 3 PC explain most of the variance (>95%) in this problem

# In[20]:


r = pd.DataFrame(r)
mean = r.mean()
std = r.std()
q2 = r.quantile(0.02)
q98 =r.quantile(0.98)


# In[21]:


stats = pd.DataFrame([mean,std,q2,q98])
stats = stats.set_index(pd.Index(['mean','standard dev','2% quantile','98% quantile']))
stats = stats.rename(columns={0:'0.25',1:'0.5',2:'1',3:'2',4:'3',5:'5',6:'7',7:'10',8:'20'})
                    
stats


# In[22]:


def progression_stats(r):
    mean = np.zeros(len(r))
    std= np.zeros(len(r))
    q2 = np.zeros(len(r))
    q98 = np.zeros(len(r))
    
    for i in range(len(r)):
        mean[i] = r[:i].mean()
        std[i] = r[:i].std()
        q2[i] = r[:i].quantile(0.02)
        q98[i] =r[:i].quantile(0.98)
    
    return mean,std,q2,q98


# In[23]:


mean_1y,std_1y,q2_1y,q98_1y = progression_stats(r.loc[:,2])


# In[24]:


# Plot the evolution of these statistical metrics over time for the 1Y term rates
x = np.array([n/250 for n in range (len(r))])
plt.plot(x,mean_1y)
plt.plot(x,std_1y)
plt.plot(x,q2_1y)
plt.plot(x,q98_1y)
plt.xlabel('Year')
plt.ylabel('value of stats')
plt.title('Progression statistics for 1yr interest rate')
plt.legend(['mean','standard dev','2% quantile','98% quantile'],loc='best')
plt.show()


# In[25]:


mean_10y,std_10y,q2_10y,q98_10y = progression_stats(r.loc[:,7])


# In[26]:


# Plot the evolution of these statistical metrics over time for the 10Y term rates
x = np.array([n/250 for n in range (len(r))])
plt.plot(x,mean_10y)
plt.plot(x,std_10y)
plt.plot(x,q2_10y)
plt.plot(x,q98_10y)
plt.xlabel('Year')
plt.ylabel('value of stats')
plt.title('Progression statistics for 10yr interest rate')
plt.legend(['mean','standard dev','2% quantile','98% quantile'],loc='best')
plt.show()


# In[27]:


mean = cmt_rates.mean()
std = cmt_rates.std()
q2 = cmt_rates.quantile(0.02)
q98 =cmt_rates.quantile(0.98)

stats_h = pd.DataFrame([mean,std,q2,q98])
stats_h = stats_h.set_index(pd.Index(['mean','standard dev','2% quantile','98% quantile']))
stats_h = stats_h.rename(columns={0:'0.25',1:'0.5',2:'1',3:'2',4:'3',5:'5',6:'7',7:'10',8:'20'})
                    
stats_h


# In[28]:


print('historical 1yr 2% quantile is: ',stats_h.loc['2% quantile','1'])
print('simulated 1yr 2% quantile is: ',stats.loc['2% quantile','1'])
print('historical 1yr 98% quantile is: ',stats_h.loc['98% quantile','1'])
print('simulated 1yr 98% quantile is: ',stats.loc['98% quantile','1'])


# In[29]:


print('historical 10yr 2% quantile is: ',stats_h.loc['2% quantile','10'])
print('simulated 10yr 2% quantile is: ',stats.loc['2% quantile','10'])
print('historical 10yr 98% quantile is: ',stats_h.loc['98% quantile','10'])
print('simulated 10yr 98% quantile is: ',stats.loc['98% quantile','10'])


# The simulated quantiles are close to but not exactly equal to historical quantiles. 
# It can be seen that simulated path have lower volatility than historical path, this may due to in our model we assume vol is constant and it is actually not in real world

# To make the simulated path more realistic, at least we should model volatility as stochastic as well.
# 
# In our case, I think to apply PCA to the covariance (or correlation) matrix of the changes is better. Because we are more interested in measuring the correlation between changes in the interest rate than measuring the interest rate itself. But that's not to say that it is unreasonable to apply PCA to levels. It is possible to apply PCA on levels and consider the eigenvectors as potentially cointegrating vectors. So we can construct factors and apply cointegration tests to the factors and include them in an ECM if significant.
