import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d

#####################################################
#Creating a univariate normal distribution with mean and standard deviation
def univariateNormalDistribution():
    mean, standardDeviation = 0, 0.5
    uniNormalDist = np.random.normal(mean,standardDeviation,100)
    # print uniNormalDist
    return uniNormalDist

def drawUniNormalDistribution():
    uniNormalDist = univariateNormalDistribution()
    mean = np.mean(uniNormalDist)
    sigma = np.std(uniNormalDist,ddof=1)
    # Display the histogram of the samples, along with the probability density function:

    count, bins, ignored = plt.hist(uniNormalDist, 40, normed=True)

    plt.plot(bins, 1 / (sigma * np.sqrt(2 * np.pi)) *np.exp( - (bins - mean)**2 / (2 * sigma**2) ),linewidth=2, color='r')
    plt.show()

# print univariateNormalDistribution()
# drawUniNormalDistribution()
#############################################################


#Creating a multivariate sample class with gaussian distribution with mean and covariance
def multivariateGaussianDistribution():
    mean_a = np.array([0,0,0])
    coVariance_a = np.array([[1,0,0],[0,1,0],[0,0,1]])
    multivariateNormalDist_a = np.random.multivariate_normal(mean_a,coVariance_a,20).T

    mean_b = np.array([1,1,1])
    coVariance_b = np.array([[1,0,0],[0,1,0],[0,0,1]])
    multivariateNormalDist_b = np.random.multivariate_normal(mean_b,coVariance_b,20).T

    return multivariateNormalDist_a,multivariateNormalDist_b


def drawMultivariateGaussianDistribution():
    sampleClass_a,sampleClass_b = multivariateGaussianDistribution()
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111, projection='3d')
    plt.rcParams['legend.fontsize'] =10
    ax.plot(sampleClass_a[0,:],sampleClass_a[1,:],sampleClass_a[2,:],'o',markersize =8, color ='blue', alpha =0.5, label='sampleClass_a')
    ax.plot(sampleClass_b[0, :], sampleClass_b[1, :], sampleClass_b[2, :], '*', markersize=8, color='red', alpha=0.5,label='sampleClass_b')

    plt.show()



print multivariateGaussianDistribution()
drawMultivariateGaussianDistribution()

