
# coding: utf-8
import numpy as np
import csv
import pandas as pd
import pickle
import RomObject

#tf.compat.v1.disable_eager_execution()
#os.environ["CUDA_VISIBLE_DEVICES"]='0,1'
#config = tf.ConfigProto()
#config.gpu_options.per_process_gpu_memory_fraction =1.0
#config.gpu_options.allow_growth = True
#session = tf.InteractiveSession(config=config)

#initiate variables
noCol = 11
noConcernVar = 4
numd = 20 
initial_time = 203
inc_time = 4 
zone1_i = 689
zone1_j = 145
glayer = 100
dt = 0.1
wt = 1000
d_inf = 1.225
U_inf = 0.005*343
sim_data_path = "~/AIRFOIL/Unsteady/Eppler387/sol01_RANS3/" #"../Data/airfoil_unsteady/"
res_data_path = "../Data/airfoil_unsteady/results/"
Tecplot_header_in = "variables=X, Y, Z, Rho, U, V, W, P, T, Vor, Qcri"
Tecplot_header_out = "variables=X, Y, Rho, U, V, P"
#create time_configurations
designs = [list(range(203, 280, 4))]
print(designs)

#pickle the designs
with open('designs4.pkl', 'wb') as output:
    pickle.dump(designs, output, pickle.HIGHEST_PROTOCOL)

#read extracted array
saved_npz = np.load("./PODarray4.npz")
snapshot_data = saved_npz['snapshot']
shp = snapshot_data.shape
xy = saved_npz['xy']
print(snapshot_data.shape)
np.savetxt("./xy.csv", xy)

#mean center the data
mean_array = None
mean_data_tensor = np.mean(snapshot_data, axis=1)
print(mean_data_tensor)
mean_centered_data = np.subtract(snapshot_data, np.tile(mean_data_tensor, (numd,1)).T)

#Singular Value Decomposition 
u, s, v = np.linalg.svd(mean_centered_data, compute_uv=True, full_matrices=False)
print(mean_centered_data.shape)
print(u.shape)
print(v.shape)
print(s.shape)

# POD coefficients
compute_coeffs = np.matmul(np.transpose(mean_centered_data),u)
e = np.sum(s)
s_energy = np.divide(s,e)*100
coeffs = compute_coeffs
mean_array = mean_data_tensor


print('Raw s-matrix')
print(s)
print('Cumulative Energy')
print(e)
print('Normalized Energy')
print(s_energy)

#some assertions for correctness
#UU^T = I
uut = np.matmul(np.transpose(u),u)
np.testing.assert_almost_equal(uut, np.eye(u.shape[1],u.shape[1]))
#VV^T = I
vvt = np.matmul(np.transpose(v),v)
np.testing.assert_almost_equal(uut, np.eye(v.shape[1],v.shape[1]))

print(coeffs.shape)
print(coeffs.dtype)
print(compute_coeffs)

#Save coefficients (coeffs) , left-singularnvectors (u) and singular values (s)
rom_object = RomObject.romobject(u, s_energy, coeffs, mean_array)
with open('rom-object4.pkl', 'wb') as output:
    pickle.dump(rom_object, output, pickle.HIGHEST_PROTOCOL)

print(mean_array.shape[0])
mean_array2 = mean_array.reshape(-1,noConcernVar)
p3d_result = np.hstack((xy, mean_array2))
#SAVE 
#np.savetxt("Case_flo_test.dat", p3d_result, delimiter=" ", header="variables = X, Y, rh, u, v, p \n zone i="+str(zone1_i)+", j="+str(zone1_j), comments=' ')
