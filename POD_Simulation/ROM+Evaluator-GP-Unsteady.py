
# coding: utf-8
#Notebook to load data from the ROM creation and evaluate 
#based on user input

import tensorflow.compat.v1 as tf
import numpy as np
import pickle
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
import itertools

tf.compat.v1.disable_eager_execution()

def kernel_function(x, y, sigma_f=1, l=1):
    """Define squared exponential kernel function."""
    kernel = sigma_f * np.exp(- (np.linalg.norm(x - y)**2) / (2 * l**2))
    return kernel

def compute_cov_matrices(x, x_star, sigma_f=1, l=1):
    """
    Compute components of the covariance matrix of the joint distribution.
    We follow the notation:
        - K = K(X, X) 
        - K_star = K(X_*, X)
        - K_star2 = K(X_*, X_*)
    """
    n = x.shape[0]
    n_star = x_star.shape[0]
    K = [kernel_function(i, j, sigma_f=sigma_f, l=l) for (i, j) in itertools.product(x, x)]
    K = np.array(K).reshape(n, n)    
    K_star2 = [kernel_function(i, j, sigma_f=sigma_f, l=l) for (i, j) in itertools.product(x_star, x_star)]
    K_star2 = np.array(K_star2).reshape(n_star, n_star)    
    K_star = [kernel_function(i, j, sigma_f=sigma_f, l=l) for (i, j) in itertools.product(x_star, x)]
    K_star = np.array(K_star).reshape(n_star, n)    
    return (K, K_star2, K_star)

def compute_gpr_parameters(y, K, K_star2, K_star, sigma_n):
    """Compute gaussian regression parameters."""
    n = K.shape[0]
    # Mean.
    f_bar_star = np.dot(K_star, np.dot(np.linalg.inv(K + (sigma_n**2)*np.eye(n)), y.reshape([n, d])))
    # Covariance.
    cov_f_star = K_star2 - np.dot(K_star, np.dot(np.linalg.inv(K + (sigma_n**2)*np.eye(n)), K_star.T))
    
    return (f_bar_star, cov_f_star)

#perform coefficient interpolation here, using numpy for it
total_steps = 20
#input_design = [251 + x for x in range(total_steps)]
input_design = np.arange(201, 241, 2)
noConcernVar = 4
zone1_i = 689
zone1_j = 145

#READ IN THE DESIGN INFORMATION
with open('designs.pkl', 'rb') as input:
    read_designs = pickle.load(input)[0]
read_designs = np.array(read_designs)
#read in saved rom object
with open('rom-object.pkl', 'rb') as input:
    read_rom = pickle.load(input)

#read xy-coordinates 
pd_data = pd.read_csv('xy.csv', dtype='float64', delimiter=' ', header=None, skipinitialspace=True)
xydata = pd_data.values
print(read_designs)
print(read_designs.shape)

with tf.device('/cpu:0'):
	coeffs = read_rom.coeffsmat
	print(coeffs[0])
	u = read_rom.umat
	mean_data = read_rom.mean_data[:,None]
	mean_tensor = tf.constant(mean_data, name="mean_data_tensor")
	# Coefficients calculated by Gaussian Process
	sigma_f = 1
	sigma_n = 0.4
	l = 1
	n_star = read_designs.shape[0]
	d = 1
	#K1 = [kernel_function(i, j, sigma_f=sigma_f, l=l) for (i, j) in itertools.product(read_designs, read_designs)]
	K, K_star2, K_star = compute_cov_matrices(read_designs, read_designs, sigma_f=sigma_f, l=l)
	#for i in range(0,n_star):
	for i, x in zip(range(total_steps), input_design):
		# Sample from prior distribution. 
		coeffs_pri = np.random.multivariate_normal(mean=coeffs[2*i], cov=K) #np.zeros(n_star)
		print("==============",coeffs_pri)
		# Compute posterior mean and covariance. 
		f_bar_star, cov_f_star = compute_gpr_parameters(coeffs_pri, K, K_star2, K_star, sigma_n)
		temp = np.zeros(n_star)
		print("=======", f_bar_star.squeeze().shape)
		#for i in range(0, n_star):
		#	# Sample from posterior distribution. 
		#	coeff = np.random.multivariate_normal(mean=f_bar_star.squeeze(), cov=cov_f_star)#.squeeze()
		#	temp += coeff
		#coeffs_pos = temp/n_star
		coeffs_sam = np.random.multivariate_normal(mean=f_bar_star.squeeze(), cov=cov_f_star, size=n_star)
		coeffs_pos = np.apply_over_axes(func=np.mean, a=coeffs_sam, axes=0).squeeze() 
		coeffs_pos_sd = np.apply_over_axes(func=np.std, a=coeffs_sam, axes=0).squeeze()
		np.savez("./coeffs.npz", t=input_design, mean=coeffs_pos, sd=coeffs_pos_sd)
		int_coeff_tensor = tf.Variable(coeffs_pos)
		#add a dim to make it a 2-D tensor
		int_coeff_tensor = tf.expand_dims(int_coeff_tensor, 0)
		print(int_coeff_tensor)
		#compute the POD approximation
		with tf.Session() as sess:
			sess.run(tf.global_variables_initializer())
			result_op = tf.matmul(int_coeff_tensor,tf.transpose(u))
			modal_result = sess.run(tf.transpose(result_op))
			modal_result.flatten()
			#mean_tensor.eval()
			sess.run(mean_tensor)
			result_op = tf.add( modal_result, mean_tensor)
			result = sess.run(result_op)
		result = result.reshape(-1, noConcernVar)
		height, width = result.shape
		#Create p3d field data
		p3d_result = np.hstack((xydata, result))
		#Save tecplot field data
		np.savetxt("./PODresults_2/Case_flo_POD_t="+str(x)+".dat", p3d_result, delimiter=" ", header="variables = X, Y, rho, u, v, p \n zone i= "+str(zone1_i)+", j="+str(zone1_j), comments=' ')

#    # Coefficients calculated by Linear interpolation
#	for i, x in zip(range(total_steps), input_design):
#		print(i)
#		hi_idx = [idx for idx,v in enumerate(read_designs) if v > x][0]
#		lo_idx = hi_idx - 1
#        #interpolate coefficients
#		interp_coeffs = coeffs[lo_idx]+(coeffs[hi_idx]-coeffs[lo_idx])*(x-read_designs[lo_idx])/(read_designs[hi_idx]-read_designs[lo_idx])
#		int_coeff_tensor = tf.Variable(interp_coeffs)
#        #add a dim to make it a 2-D tensor
#		int_coeff_tensor = tf.expand_dims(int_coeff_tensor, 0)
#		print(int_coeff_tensor)       
#        #compute the POD approximation
#		with tf.Session() as sess:
#			sess.run(tf.global_variables_initializer())
#			result_op = tf.matmul(int_coeff_tensor,tf.transpose(u))
#			modal_result = sess.run(tf.transpose(result_op))
#			modal_result.flatten()
#			#mean_tensor.eval()
#			sess.run(mean_tensor)
#			result_op = tf.add( modal_result, mean_tensor)
#			result = sess.run(result_op)
#		result = result.reshape(-1, noConcernVar)
#		height, width = result.shape
#    #Create p3d field data
#		p3d_result = np.hstack((xydata, result))
#			#Save tecplot field data
#		np.savetxt("./PODresults/Case_flo_POD_(t="+str(x)+").dat", p3d_result, delimiter=" ", header="variables = X, Y, rho, u, v, p \n zone i= "+str(zone1_i)+", j="+str(zone1_j), comments=' ')
