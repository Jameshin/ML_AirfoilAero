#"""
#@author: Maziar Raissi
#"""

import tensorflow.compat.v1 as tf
import numpy as np
import scipy.io
import time
import sys
import os
import pandas as pd
import pickle
import RomObject

from CFDFunctions import neural_net, Euler_uComp_2D, Gradient_Velocity_2D, \
                      tf_session, mean_squared_error, relative_error

tf.compat.v1.disable_eager_execution()

class HFM(object):
    # notational conventions
    # _tf: placeholders for input/output data and points used to regress the equations
    # _pred: output of neural network
    # _eqns: points used to regress the equations
    # _data: input-output data
    # _inlet: input-output data at the inlet
    # _star: preditions
    
    def __init__(self, t_pod_data, t_pod_eqns, t_data, x_data, y_data, 
                 d_data, u_data, v_data, p_data, phi, mean_data, a0_data, 
                 a1_data, a2_data, a3_data, a4_data, a5_data, a6_data, a7_data, 
                 a8_data, a9_data, a10_data,
                 a11_data, a12_data, a13_data, a14_data, a15_data, a16_data, a17_data,
                 a18_data, a19_data, t_eqns, x_eqns, y_eqns, layers, batch_size,
                 Pec, Rey):
        
        # specs
        self.layers = layers
        self.batch_size = batch_size
        
        # flow properties
        self.Pec = Pec
        self.Rey = Rey

        # base space
        self.phi = phi
        self.mean_data = mean_data
        noConcernsVar = 4 
        N = x_data.shape[0]
        T = t_pod_data.shape[0]
        # data
        #[self.t_pod_data, self.t_pod_eqns, self.a0_data, self.a1_data, self.a2_data, self.a3_data, self.a4_data, self.a5_data, self.a6_data, self.a7_data, self.a8_data, self.a9_data, self.a10_data, self.a11_data, self.a12_data, self.a13_data, self.a14_data, self.a15_data, self.a16_data, self.a17_data, self.a18_data, self.a19_data] = [t_pod_data, t_pod_eqns, a0_data, a1_data, a2_data, a3_data, a4_data, a5_data, a6_data, a7_data, a8_data, a9_data, a10_data, a11_data, a12_data, a13_data, a14_data, a15_data, a16_data, a17_data, a18_data, a19_data]
        [self.t_data, self.x_data, self.y_data, self.d_data, self.u_data, self.v_data, self.p_data] = [t_data, x_data, y_data, d_data, u_data, v_data, p_data]
        #self.a_data = tf.concat([self.a0_data, self.a1_data,
        #                            self.a2_data,
        #                            self.a3_data, self.a4_data,
        #                            self.a5_data, self.a6_data,
        #                            self.a7_data, self.a8_data,
        #                            self.a9_data, self.a10_data,
        #                            self.a11_data, self.a12_data,
        #                            self.a13_data, self.a14_data,
        #                            self.a15_data, self.a16_data,
        #                            self.a17_data, self.a18_data,
        #                            self.a19_data], axis=1) 
        # placeholders
        [self.t_pod_data_tf, self.d_data_tf, self.u_data_tf, self.v_data_tf, self.p_data_tf, self.a0_data_tf, self.a1_data_tf, self.a2_data_tf, self.a3_data_tf, self.a4_data_tf, self.a5_data_tf, self.a6_data_tf, self.a7_data_tf, self.a8_data_tf, self.a9_data_tf, self.a10_data_tf, self.a11_data_tf, self.a12_data_tf, self.a13_data_tf, self.a14_data_tf, self.a15_data_tf, self.a16_data_tf, self.a17_data_tf, self.a18_data_tf, self.a19_data_tf] = [tf.placeholder(tf.float64, shape=[None, 1]) for _ in range(25)]
        [self.t_pod_eqns_tf, self.t_data_tf, self.x_data_tf, self.y_data_tf] = [tf.placeholder(tf.float64, shape=[None, 1]) for _ in range(4)]

        # physics "uninformed" neural networks
        #self.net_pod= neural_net(self.t_pod_data, self.t_pod_data, self.t_pod_data, self.t_pod_data, self.t_pod_data, self.t_pod_data, self.t_pod_data, self.t_pod_data, self.t_pod_data, self.t_pod_data, self.t_pod_data, self.t_pod_data, self.t_pod_data, self.t_pod_data, self.t_pod_data, layers = self.layers)
        self.net_duvp = neural_net(self.t_data, self.x_data, self.y_data, layers = self.layers) #[3,12,12,12,12,12,12,12,4])
        #print(np.array(self.t_pod_data).shape)
        
        #[self.a0_data_pred, self.a1_data_pred, self.a2_data_pred, 
        # self.a3_data_pred, self.a4_data_pred, self.a5_data_pred, 
        # self.a6_data_pred, self.a7_data_pred, self.a8_data_pred, 
        # self.a9_data_pred, self.a10_data_pred,
        # self.a11_data_pred, self.a12_data_pred,
        # self.a13_data_pred, self.a14_data_pred,
        # self.a15_data_pred, self.a16_data_pred,
        # self.a17_data_pred, self.a18_data_pred, 
        # self.a19_data_pred] = self.net_pod(self.t_pod_data, self.t_pod_data, self.t_pod_data, self.t_pod_data, self.t_pod_data, self.t_pod_data, self.t_pod_data_tf, self.t_pod_data_tf, self.t_pod_data_tf, self.t_pod_data_tf, self.t_pod_data_tf, self.t_pod_data_tf, self.t_pod_data_tf, self.t_pod_data_tf, self.t_pod_data_tf)
                
        #self.a_data_pred = tf.concat([self.a0_data_pred, self.a1_data_pred, 
        #                             self.a2_data_pred,
        #                             self.a3_data_pred, self.a4_data_pred,
        #                             self.a5_data_pred, self.a6_data_pred,
        #                             self.a7_data_pred, self.a8_data_pred,
        #                             self.a9_data_pred, self.a10_data_pred,
        #                             self.a11_data_pred, self.a12_data_pred,
        #                             self.a13_data_pred, self.a14_data_pred,
        #                             self.a15_data_pred, self.a16_data_pred,
        #                             self.a17_data_pred, self.a18_data_pred,
        #                             self.a19_data_pred], axis=1)

        #self.a_data_tf = tf.concat([self.a0_data_tf, self.a1_data_tf,
        #                            self.a2_data_tf,
        #                            self.a3_data_tf, self.a4_data_tf,
        #                            self.a5_data_tf, self.a6_data_tf,
        #                            self.a7_data_tf, self.a8_data_tf,
        #                            self.a9_data_tf, self.a10_data_tf,
        #                            self.a11_data_tf, self.a12_data_tf,
        #                            self.a13_data_tf, self.a14_data_tf,
        #                            self.a15_data_tf, self.a16_data_tf,
        #                            self.a17_data_tf, self.a18_data_tf,
        #                            self.a19_data_tf], axis=1) 
        # physics "informed" neural networks
        [self.d_data_pred, self.u_data_pred,
         self.v_data_pred,
         self.p_data_pred] = self.net_duvp(self.t_data_tf,
                                           self.x_data_tf,
                                           self.y_data_tf)

        #[self.a0_eqns_pred, self.a1_eqns_pred, self.a2_eqns_pred,
        # self.a3_eqns_pred, self.a4_eqns_pred,
        # self.a5_eqns_pred, self.a6_eqns_pred,
        # self.a7_eqns_pred, self.a8_eqns_pred,
        # self.a9_eqns_pred, self.a10_eqns_pred,
        # self.a11_eqns_pred, self.a12_eqns_pred,
        # self.a13_eqns_pred, self.a14_eqns_pred,
        # self.a15_eqns_pred, self.a16_eqns_pred,
        # self.a17_eqns_pred, self.a18_eqns_pred,
        # self.a19_eqns_pred] = self.net_pod(self.t_pod_eqns_tf)

        #self.a_eqns_pred = tf.concat([self.a0_eqns_pred, self.a1_eqns_pred, 
        #                              self.a2_eqns_pred,
        #                             self.a3_eqns_pred, self.a4_eqns_pred,
        #                             self.a5_eqns_pred, self.a6_eqns_pred,
        #                             self.a7_eqns_pred, self.a8_eqns_pred,
        #                             self.a9_eqns_pred, self.a10_eqns_pred,
        #                             self.a11_eqns_pred, self.a12_eqns_pred,
        #                             self.a13_eqns_pred, self.a14_eqns_pred,
        #                             self.a15_eqns_pred, self.a16_eqns_pred,
        #                             self.a17_eqns_pred, self.a18_eqns_pred,
        #                             self.a19_eqns_pred], axis=1)
        
        #U_pod_pred = tf.add(tf.transpose(tf.matmul(self.a_data_pred, tf.transpose(tf.constant(phi, tf.float64)))), tf.tile(tf.constant(mean_data, tf.float64), tf.constant([1,T], tf.int32)))
        #for i in range(Ntime):
        #    temp = tf.reshape(U_pod_pred[:,i], [-1, noConcernVar])
        #    if i == 0:
        #        U_pred = temp
        #    else:
        #        U_pred = tf.concat([U_pred, temp], axis=0)
        #self.d_data_pred2 = U_pred[:,0][:,None]
        #self.u_data_pred2 = U_pred[:,1][:,None]
        #self.v_data_pred2 = U_pred[:,2][:,None]
        #self.p_data_pred2 = U_pred[:,3][:,None]
        #[self.e1_eqns_pred,
        # self.e2_eqns_pred,
        # self.e3_eqns_pred] = Euler_uIncomp_POD(self.a_eqns_pred, self.phi, self.mean_data,
        #                                       self.t_pod_eqns_tf,
        #                                       self.x_eqns_tf,
        #                                       self.y_eqns_tf,
        #                                       self.Pec,
        #                                       self.Rey)
        [self.e1_data_pred,
         self.e2_data_pred,
         self.e3_data_pred, 
         self.e4_data_pred] = Euler_uComp_2D(self.d_data_pred, 
                                               self.u_data_pred,
                                               self.v_data_pred,
                                               self.p_data_pred,
                                               self.t_data_tf,
                                               self.x_data_tf,
                                               self.y_data_tf,
                                               self.Pec,
                                               self.Rey)
        
        # gradients required for the lift and drag forces
        #[self.u_x_eqns_pred,
        # self.v_x_eqns_pred,
        # self.u_y_eqns_pred,
        # self.v_y_eqns_pred] = Gradient_Velocity_2D(self.u_eqns_pred,
        #                                            self.v_eqns_pred,
        #                                            self.x_eqns_tf,
        #                                            self.y_eqns_tf)
        
        # loss
        #u_x = tf.gradients(self.u_data_pred, self.x_data)
        #v_y = tf.gradients(self.v_data_pred, self.y_data)
        #e1 = u_x + v_y
        #print(u_x, v_y)
        self.loss = mean_squared_error(self.d_data_pred, self.d_data_tf) + \
                    mean_squared_error(self.u_data_pred, self.u_data_tf) + \
                    mean_squared_error(self.v_data_pred, self.v_data_tf) + \
                    mean_squared_error(self.p_data_pred, self.p_data_tf) + \
                    mean_squared_error(self.e1_data_pred, 0.0) + \
                    mean_squared_error(self.e2_data_pred, 0.0) + \
                    mean_squared_error(self.e3_data_pred, 0.0) + \
                    mean_squared_error(self.e4_data_pred, 0.0)
        
        # optimizers
        self.learning_rate = tf.placeholder(tf.float32, shape=[])
        self.optimizer = tf.train.AdamOptimizer(learning_rate = self.learning_rate)
        self.train_op = self.optimizer.minimize(self.loss)
        
        self.sess = tf_session()
    
    def train(self, total_time, learning_rate):
        
        N_data = self.t_data.shape[0]
        #N_eqns = self.t_eqns.shape[0]
        
        start_time = time.time()
        running_time = 0
        it = 0
        while running_time < total_time:
            
            idx_data = np.random.choice(N_data, min(self.batch_size, N_data))
            #idx_eqns = np.random.choice(N_eqns, min(self.batch_size, N_eqns))
            if it == 4000:
                learning_rate = 1e-3
            if it == 12000:
                learning_rate = 1e-4
            if it == 20000:
                learning_rate = 1e-5
            if it == 50000:
                learning_rate = 1e-7
            (t_data_batch,
             x_data_batch,
             y_data_batch,
             d_data_batch, u_data_batch, 
             v_data_batch, p_data_batch) = (self.t_data[idx_data,:],
                              self.x_data[idx_data,:],
                              self.y_data[idx_data,:],
                              self.d_data[idx_data,:],
                              self.u_data[idx_data,:],
                              self.v_data[idx_data,:],
                              self.p_data[idx_data,:])
 
            #(t_pod_data_batch,
            # a0_data_batch, a1_data_batch, a2_data_batch, a3_data_batch, 
            # a4_data_batch, a5_data_batch, a6_data_batch,
            # a7_data_batch, a8_data_batch, a9_data_batch,
            # a10_data_batch, a11_data_batch, a12_data_batch,
            # a13_data_batch, a14_data_batch, a15_data_batch,
            # a16_data_batch, a17_data_batch, a18_data_batch,
            # a19_data_batch) = (self.t_pod_data[idx_data,:],
            #                    self.a0_data, self.a1_data, 
            #                    self.a2_data, 
            #                    self.a3_data, self.a4_data,
            #                    self.a5_data, self.a6_data,
            #                    self.a7_data, self.a8_data,
            #                    self.a9_data, self.a10_data,
            #                    self.a11_data, self.a12_data,
            #                    self.a13_data, self.a14_data,
            #                    self.a15_data, self.a16_data,
            #                    self.a17_data, self.a18_data,
            #                    self.a19_data)

            #t_pod_eqns_batch = self.t_pod_eqns[idx_eqns]

            tf_dict = {self.t_data_tf: t_data_batch,
                       self.x_data_tf: x_data_batch,
                       self.y_data_tf: y_data_batch,
                       self.u_data_tf: u_data_batch,
                       self.v_data_tf: v_data_batch,
                       self.d_data_tf: d_data_batch,
                       self.p_data_tf: p_data_batch,
                       self.learning_rate: learning_rate}
            #tf_dict = {self.t_pod_data_tf: t_pod_data_batch,
            #           self.a0_data_tf: a0_data_batch, 
            #           self.a1_data_tf: a1_data_batch, self.a2_data_tf: a2_data_batch,
            #           self.a3_data_tf: a3_data_batch, self.a4_data_tf: a4_data_batch,
            #           self.a5_data_tf: a5_data_batch, self.a6_data_tf: a6_data_batch,
            #           self.a7_data_tf: a7_data_batch, self.a8_data_tf: a8_data_batch,
            #           self.a9_data_tf: a9_data_batch, self.a10_data_tf: a10_data_batch,
            #           self.a11_data_tf: a11_data_batch, self.a12_data_tf: a12_data_batch,
            #           self.a13_data_tf: a13_data_batch, self.a14_data_tf: a14_data_batch,
            #           self.a15_data_tf: a15_data_batch, self.a16_data_tf: a16_data_batch,
            #           self.a17_data_tf: a17_data_batch, self.a18_data_tf: a18_data_batch,
            #           self.a19_data_tf: a19_data_batch, 
            #           self.t_pod_eqns_tf: t_pod_eqns_batch, self.t_data_tf:self.t_data,
            #           self.x_data_tf:self.x_data, self.y_data_tf:self.y_data, 
            #           self.u_data_tf: self.u_data, self.v_data_tf: self.v_data,
            #           self.d_data_tf: self.d_data, self.p_data_tf: self.p_data,
            #           self.learning_rate: learning_rate}
            
            self.sess.run([self.train_op], tf_dict)
            
            # Print
            if it % 10 == 0:
                elapsed = time.time() - start_time
                running_time += elapsed/3600.0
                [loss_value,
                 learning_rate_value] = self.sess.run([self.loss,
                                                       self.learning_rate], tf_dict)
                print('It: %d, Loss: %.3e, Time: %.2fs, Running Time: %.2fh, Learning Rate: %.1e'
                      %(it, loss_value, elapsed, running_time, learning_rate_value))
                sys.stdout.flush()
                start_time = time.time()
            it += 1
    
    def predict(self, t_star, x_star, y_star):

        tf_dict = {self.t_data_tf: t_star, self.x_data_tf: x_star, self.y_data_tf: y_star}

        d_star = self.sess.run(self.d_data_pred, tf_dict)
        u_star = self.sess.run(self.u_data_pred, tf_dict)
        v_star = self.sess.run(self.v_data_pred, tf_dict)
        p_star = self.sess.run(self.p_data_pred, tf_dict)

        return d_star, u_star, v_star, p_star 
    #def predict(self, t_star):
        
        #tf_dict = {self.t_pod_data_tf: t_star}
        
        #a0_star = self.sess.run(self.a0_data_pred, tf_dict)
        #a1_star = self.sess.run(self.a1_data_pred, tf_dict)
        #a2_star = self.sess.run(self.a2_data_pred, tf_dict)
        #a3_star = self.sess.run(self.a3_data_pred, tf_dict)
        #a4_star = self.sess.run(self.a4_data_pred, tf_dict)
        #a5_star = self.sess.run(self.a5_data_pred, tf_dict)
        #a6_star = self.sess.run(self.a6_data_pred, tf_dict)
        #a7_star = self.sess.run(self.a7_data_pred, tf_dict)
        #a8_star = self.sess.run(self.a8_data_pred, tf_dict)
        #a9_star = self.sess.run(self.a9_data_pred, tf_dict)
        #a10_star = self.sess.run(self.a10_data_pred, tf_dict)
        #a11_star = self.sess.run(self.a11_data_pred, tf_dict)
        #a12_star = self.sess.run(self.a12_data_pred, tf_dict)
        #a13_star = self.sess.run(self.a13_data_pred, tf_dict)
        #a14_star = self.sess.run(self.a14_data_pred, tf_dict)
        #a15_star = self.sess.run(self.a15_data_pred, tf_dict)
        #a16_star = self.sess.run(self.a16_data_pred, tf_dict)
        #a17_star = self.sess.run(self.a17_data_pred, tf_dict)
        #a18_star = self.sess.run(self.a18_data_pred, tf_dict)
        #a19_star = self.sess.run(self.a19_data_pred, tf_dict)
        #a_star = np.hstack((a0_star, a1_star,  a2_star, a3_star, a4_star, a5_star,
        #               a6_star,  a7_star, a8_star, a9_star, a10_star,
        #               a11_star,  a12_star, a13_star, a14_star, a15_star,
        #               a16_star,  a17_star, a18_star,
        #               a19_star))
        
        #return a_star

if __name__ == "__main__":
    with tf.device('/gpu:0'):
        batch_size = 10000 
        layers = [3] + 10*[4*10] + [4]
    
        # Load Data
        sim_data_path = "~/AIRFOIL/Unsteady/Eppler387/sol01_RANS3/"
        # create directory if not exist
        #os.makedirs(os.path.dirname(res_data_path), exist_ok=True)
        # list of file names
        filenames = []
        merged = []
        Ntime = 0
        numd = 20
        initial_time = 201
        inc_time = 4
        dt = 0.1
        for i in range(0, numd):
            filenames.append(sim_data_path+"flo001.0000"+str(initial_time+i*inc_time).rjust(3,'0')+"uns")
            Ntime += 1
        #print(Ntime, filenames)
        t_star = np.arange(Ntime)*dt # 1xT(=1)
        ###
        #perform coefficient interpolation here, using numpy for it
        total_steps = 20
        #input_design = [251 + x for x in range(total_steps)]
        input_times = np.arange(203, 280, 4)*dt
        noConcernVar = 4
        zone1_i = 689
        zone1_j = 145

        #READ IN THE POD DESIGN INFORMATION
        with open('./designs3.pkl', 'rb') as input:
            read_times = pickle.load(input)[0]
        read_times = np.array(read_times)*dt
        #read in saved rom object
        with open('./rom-object3.pkl', 'rb') as input:
            read_rom = pickle.load(input)
        ###
        #read xy-coordinates
        pd_data = pd.read_csv('./xy.csv', dtype='float64', delimiter=' ', header=None, skipinitialspace=True)
        xydata = pd_data.values

        coeffs = np.array(read_rom.coeffsmat)
        phi = np.array(read_rom.umat)
        mean_data = np.array(read_rom.mean_data[:,None])
        #mean_tensor = tf.constant(mean_data, name="mean_data_tensor")
        U_pod = np.add(np.transpose(np.matmul(coeffs, np.transpose(phi))), 
                        np.tile(mean_data, (1,Ntime)))
        for i in range(Ntime):
            U_mean = mean_data.reshape(-1, noConcernVar)
            U_phi = phi[:,i].reshape(-1, noConcernVar)
            U = U_pod[:,i].reshape(-1, noConcernVar)
            d_mean = U_mean[:,0][:,None]
            u_mean = U_mean[:,1][:,None]
            v_mean = U_mean[:,2][:,None]
            p_mean = U_mean[:,3][:,None]
            d_phi = U_phi[:,0][:,None]
            u_phi = U_phi[:,1][:,None]
            v_phi = U_phi[:,2][:,None]
            p_phi = U_phi[:,3][:,None]
            d_full = U[:,0][:,None]
            u_full = U[:,1][:,None]
            v_full = U[:,2][:,None]
            p_full = U[:,3][:,None]
            p3d_result = np.hstack((xydata[:,0][:,None], xydata[:,1][:,None], d_full, u_full, v_full, p_full))
            p3d_mode = np.hstack((xydata[:,0][:,None], xydata[:,1][:,None], d_phi, u_phi, v_phi, p_phi))
            np.savetxt("./PODfield_RANS/Case_flo_t="+str(i)+".dat", p3d_result, delimiter=" ", header="variables = X, Y, c, u, v, p \n zone i="+str(zone1_i)+" j="+str(zone1_j)+" ", comments=' ')
            np.savetxt("./PODfield_RANS/Mode_"+str(i)+".dat", p3d_mode, delimiter=" ", header="variables = X, Y, c, u, v, p \n zone i="+str(zone1_i)+" j="+str(zone1_j)+" ", comments=' ')
p3d_mean = np.hstack((xydata[:,0][:,None], xydata[:,1][:,None], d_mean, u_mean, v_mean, p_mean))
np.savetxt("./PODfield_RANS/Mean_field_t="+str(i)+".dat", p3d_mean, delimiter=" ", header="variables = X, Y, c, u, v, p \n zone i="+str(zone1_i)+" j="+str(zone1_j)+" ", comments=' ')
