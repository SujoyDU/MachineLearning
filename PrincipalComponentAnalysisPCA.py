#http://sebastianraschka.com/Articles/2014_pca_step_by_step.html
# For the following example, we will generate 40 3-dimensional samples randomly drawn from a multivariate Gaussian distribution.
# Here, we will assume that the samples stem from two different classes, u1 and u2


# np.random.seed(234234782384239784) # random seed for consistency

# A reader pointed out that Python 2.7 would raise a
# "ValueError: object of too small depth for desired array".
# This can be avoided by choosing a smaller random seed, e.g. 1
# or by completely omitting this line, since I just used the random seed for
# consistency.

# mu_vec1 = np.array([0,0,0])
# cov_mat1 = np.array([[1,0,0],[0,1,0],[0,0,1]])
# class1_sample = np.random.multivariate_normal(mu_vec1, cov_mat1, 20).T
# assert class1_sample.shape == (3,20), "The matrix has not the dimensions 3x20"
#
# mu_vec2 = np.array([1,1,1])
# cov_mat2 = np.array([[1,0,0],[0,1,0],[0,0,1]])
# class2_sample = np.random.multivariate_normal(mu_vec2, cov_mat2, 20).T
# assert class2_sample.shape == (3,20), "The matrix has not the dimensions 3x20"

import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d



#class 1
mean_vec1 = np.array([0,0,0]) #mean for class 1
cov_mat1 = np.array([[1,0,0],[0,1,0],[0,0,1]]) #covariance
# print mean_vec1
# print cov_mat1
class1_sample = np.random.multivariate_normal(mean_vec1, cov_mat1, 20) #create a (20*3) gaussian matrix
# print class1_sample.shape

class1_sample = class1_sample.T #convert to transpose matrix
# print class1_sample_tran.shape
assert class1_sample.shape == (3,20), "The matrix has not the dimensions 3x20" #check for dimension

#class 2
mean_vec2 = np.array([1,1,1]) #mean for class 2
cov_mat2 = np.array([[1,0,0],[0,1,0],[0,0,1]]) #covariance for class 2
class2_sample = np.random.multivariate_normal(mean_vec2,cov_mat2,20) #create a (20*3) gaussian matrix
# print class2_sample.shape
class2_sample = class2_sample.T #convert to transpose matrix
assert class2_sample.shape == (3,20), "The matrix has not the dimensions 3x20" #check for dimension


# fig = plt.figure(figsize=(8,8))
# ax = fig.add_subplot(111,projection = '3d')
# plt.rcParams['legend.fontsize'] = 10
# ax.plot(class1_sample[0,:],class1_sample[1,:],class1_sample[2,:], 'o', markersize=8, color= 'blue', alpha=0.5, label = 'class1')
# ax.plot(class2_sample[0,:],class2_sample[1,:],class2_sample[2,:], '^', markersize=8, color= 'green', alpha=0.5, label = 'class2')
# plt.title('Samples for class1 and class2')
# ax.legend(loc='upper right')
# plt.show()

all_samples = np.concatenate((class1_sample,class2_sample),axis=1)
assert all_samples.shape == (3,40), "The matrix has not the dimensions 3x40"

mean_x = np.mean(all_samples[0,:])
mean_y = np.mean(all_samples[1,:])
mean_z = np.mean(all_samples[2,:])

mean_vector = np.array([[mean_x],[mean_y],[mean_z]])
# print mean_vector

scatter_mat = np.zeros((3,3))
# print scatter_mat
# print all_samples[:,0].reshape(3,1)
# x1= all_samples[:,0].reshape(3,1)-mean_vector
# x2= (all_samples[:,0].reshape(3,1)-mean_vector).T
#
# print x1
# print x2
# x1 = np.array([[9],[4],[5]])
# x2 = np.array([[1,2,3]])
# print x1
# print x2
# print x1.dot(x2)0
for i in range(all_samples.shape[1]):
    scatter_mat += (all_samples[:,i].reshape(3,1)-mean_vector).dot((all_samples[:,i].reshape(3,1)-mean_vector).T)
# print scatter_mat

cov_mat = np.cov([all_samples[0,:],all_samples[1,:], all_samples[2,:]])
# print cov_mat

# eigenvectors and eigenvalues for the from the scatter matrix
eig_val_sc, eig_vec_sc = np.linalg.eig(scatter_mat)

# eigenvectors and eigenvalues for the from the covariance matrix
eig_val_cov, eig_vec_cov = np.linalg.eig(cov_mat)

print eig_val_cov
print eig_val_sc

print eig_vec_cov
print eig_vec_sc

for i in range(len(eig_val_sc)):
    eigvec_sc = eig_vec_sc[:,i].reshape(1,3).T
    eigvec_cov = eig_vec_cov[:,i].reshape(1,3).T
    assert eigvec_sc.all() == eigvec_cov.all(), 'Eigenvectors are not identical'

    # print('Eigenvector {}: \n{}'.format(i+1, eigvec_sc))
    # print('Eigenvalue {} from scatter matrix: {}'.format(i+1, eig_val_sc[i]))
    # print('Eigenvalue {} from covariance matrix: {}'.format(i+1, eig_val_cov[i]))
    # print('Scaling factor: ', eig_val_sc[i]/eig_val_cov[i])
    # print(40 * '-')

print i in range (len(eig_val_sc))