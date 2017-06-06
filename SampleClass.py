import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d
from matplotlib.patches import FancyArrowPatch

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

    drawMultivariateGaussianDistribution(multivariateNormalDist_a, multivariateNormalDist_b)
    return multivariateNormalDist_a,multivariateNormalDist_b


def drawMultivariateGaussianDistribution(sampleClass_a,sampleClass_b):
    # sampleClass_a,sampleClass_b = multivariateGaussianDistribution()
    fig = plt.figure(0,figsize=(10,10))
    ax = fig.add_subplot(111, projection='3d')
    plt.rcParams['legend.fontsize'] =10
    ax.plot(sampleClass_a[0,:],sampleClass_a[1,:],sampleClass_a[2,:],'o',markersize =8, color ='blue', alpha =0.5, label='sampleClass_a')
    ax.plot(sampleClass_b[0, :], sampleClass_b[1, :], sampleClass_b[2, :], '*', markersize=8, color='red', alpha=0.5,label='sampleClass_b')
    # plt.show()



# print multivariateGaussianDistribution()
# drawMultivariateGaussianDistribution()
#############################################################


#Calculating eigen vector and eigen value from co-variance matrix

def eigenVecVal():
    sampleClass_a, sampleClass_b = multivariateGaussianDistribution()
    sampleClass = np.concatenate((sampleClass_a,sampleClass_b),axis=1)
    cov_mat = np.cov([sampleClass[0,:],sampleClass[1,:],sampleClass[2,:]])
    print cov_mat

    eigen_val, eigen_vec = np.linalg.eig(cov_mat)

    drawEigen(sampleClass,cov_mat,eigen_val,eigen_vec)
    return sampleClass,cov_mat,eigen_val,eigen_vec

# print eigenVecVal()
#############################################################


#plotting eigen vectors and eigen value

class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
        FancyArrowPatch.draw(self, renderer)


def drawEigen(all_samples,cov_mat,eigen_val,eigen_vec):
    fig = plt.figure(figsize=(7,7))
    ax = fig.add_subplot(111, projection='3d')

    # all_samples, cov_mat, eigen_val, eigen_vec = eigenVecVal()
    mean_x = np.mean(all_samples[0,:])
    mean_y = np.mean(all_samples[1,:])
    mean_z = np.mean(all_samples[2,:])

    plt.figure(1)
    ax.plot(all_samples[0,:], all_samples[1,:], all_samples[2,:], 'o', markersize=8, color='green', alpha=0.2)
    ax.plot([mean_x], [mean_y], [mean_z], 'o', markersize=10, color='red', alpha=0.5)
    for v in eigen_vec.T:
        a = Arrow3D([mean_x, v[0]], [mean_y, v[1]], [mean_z, v[2]], mutation_scale=20, lw=3, arrowstyle="-|>", color="r")
        ax.add_artist(a)
    ax.set_xlabel('x_values')
    ax.set_ylabel('y_values')
    ax.set_zlabel('z_values')

    plt.title('Eigenvectors')

    # plt.show()

# drawEigen()
#############################################################

#Sorting Eigen vectors by decreasing eigen values

def sortEigen():
    all_samples, cov_mat, eig_val, eig_vec = eigenVecVal()

    # Make a list of (eigenvalue, eigenvector) tuples
    eig_pairs = [(np.abs(eig_val[i]), eig_vec[:, i]) for i in range(len(eig_val))]

    # Sort the (eigenvalue, eigenvector) tuples from high to low
    eig_pairs.sort(key=lambda x: x[0], reverse=True)

    # Visually confirm that the list is correctly sorted by decreasing eigenvalues
    for i in eig_pairs:
        print(i)

    matrix_w = np.hstack((eig_pairs[0][1].reshape(3, 1), eig_pairs[1][1].reshape(3, 1)))
    print('Matrix W:\n', matrix_w)

    return all_samples,matrix_w

# sortEigen()
#############################################################


# Transforming the samples into new subspace

def newSubspace():
    all_samples, matrix_w = sortEigen()
    transformed = matrix_w.T.dot(all_samples)
    assert transformed.shape == (2,40), "The matrix is not 2x40 dimensional."

    plt.figure(2)
    plt.plot(transformed[0, 0:20], transformed[1, 0:20], 'o', markersize=7, color='blue', alpha=0.5, label='class1')
    plt.plot(transformed[0, 20:40], transformed[1, 20:40], '*', markersize=7, color='red', alpha=0.5, label='class2')
    plt.xlim([-4, 4])
    plt.ylim([-4, 4])
    plt.xlabel('x_values')
    plt.ylabel('y_values')
    plt.legend()
    plt.title('Transformed samples with class labels')

    plt.show()

newSubspace()
