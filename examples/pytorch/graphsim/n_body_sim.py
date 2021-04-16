from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import numpy as np
from math import sin, cos, radians, pi
import argparse

'''
This adapted from comes from https://github.com/jsikyoon/Interaction-networks_tensorflow
which generates multi-body dynamic simulation data for Interaction network
'''

# 5 features on the state [mass,x,y,x_vel,y_vel]
fea_num = 5
# G stand for Gravity constant 10**5 can help numerical stability
G = 10**5
# time step
diff_t = 0.001

def init(total_state, n_body, fea_num, orbit):
    data = np.zeros((total_state, n_body, fea_num), dtype=float)
    if(orbit):
        data[0][0][0] = 100
        data[0][0][1:5] = 0.0
        # The position are initialized randomly.
        for i in range(1, n_body):
            data[0][i][0] = np.random.rand()*8.98+0.02
            distance = np.random.rand()*90.0+10.0
            theta = np.random.rand()*360
            theta_rad = pi/2 - radians(theta)
            data[0][i][1] = distance*cos(theta_rad)
            data[0][i][2] = distance*sin(theta_rad)
            data[0][i][3] = -1*data[0][i][2]/norm(data[0][i][1:3])*(
                G*data[0][0][0]/norm(data[0][i][1:3])**2)*distance/1000
            data[0][i][4] = data[0][i][1]/norm(data[0][i][1:3])*(
                G*data[0][0][0]/norm(data[0][i][1:3])**2)*distance/1000
    else:
        for i in range(n_body):
            data[0][i][0] = np.random.rand()*8.98+0.02
            distance = np.random.rand()*90.0+10.0
            theta = np.random.rand()*360
            theta_rad = pi/2 - radians(theta)
            data[0][i][1] = distance*cos(theta_rad)
            data[0][i][2] = distance*sin(theta_rad)
            data[0][i][3] = np.random.rand()*6.0-3.0
            data[0][i][4] = np.random.rand()*6.0-3.0
    return data

def norm(x):
    return np.sqrt(np.sum(x**2))

def get_f(reciever, sender):
    diff = sender[1:3]-reciever[1:3]
    distance = norm(diff)
    if(distance < 1):
        distance = 1
    return G*reciever[0]*sender[0]/(distance**3)*diff

# Compute stat according to the paper for normalization
def compute_stats(train_curr):
    data = np.vstack(train_curr).reshape(-1, fea_num)
    stat_median = np.median(data, axis=0)
    stat_max = np.quantile(data, 0.95, axis=0)
    stat_min = np.quantile(data, 0.05, axis=0)
    return stat_median, stat_max, stat_min

def calc(cur_state, n_body):
    next_state = np.zeros((n_body, fea_num), dtype=float)
    f_mat = np.zeros((n_body, n_body, 2), dtype=float)
    f_sum = np.zeros((n_body, 2), dtype=float)
    acc = np.zeros((n_body, 2), dtype=float)
    for i in range(n_body):
        for j in range(i+1, n_body):
            if(j != i):
                f = get_f(cur_state[i][:3], cur_state[j][:3])
                f_mat[i, j] += f
                f_mat[j, i] -= f
        f_sum[i] = np.sum(f_mat[i], axis=0)
        acc[i] = f_sum[i]/cur_state[i][0]
        next_state[i][0] = cur_state[i][0]
        next_state[i][3:5] = cur_state[i][3:5]+acc[i]*diff_t
        next_state[i][1:3] = cur_state[i][1:3]+next_state[i][3:5]*diff_t
    return next_state

# The state is [mass,pos_x,pos_y,vel_x,vel_y]* n_body
def gen(n_body, num_steps, orbit):
    # initialization on just first state
    data = init(num_steps, n_body, fea_num, orbit)
    for i in range(1, num_steps):
        data[i] = calc(data[i-1], n_body)
    return data

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--num_bodies', type=int, default=6)
    argparser.add_argument('--num_traj', type=int, default=10)
    argparser.add_argument('--steps', type=int, default=1000)
    argparser.add_argument('--data_path', type=str, default='data')

    args = argparser.parse_args()
    if not os.path.exists(args.data_path):
        os.mkdir(args.data_path)

    # Generate data
    data_curr = []
    data_next = []

    for i in range(args.num_traj):
        raw_traj = gen(args.num_bodies, args.steps, True)
        data_curr.append(raw_traj[:-1])
        data_next.append(raw_traj[1:])
        print("Train Traj: ", i)

    # Compute normalization statistic from data
    stat_median, stat_max, stat_min = compute_stats(data_curr)
    data = np.vstack(data_curr)
    label = np.vstack(data_next)[:, :, 3:5]
    shuffle_idx = np.arange(data.shape[0])
    np.random.shuffle(shuffle_idx)
    train_split = int(0.9*data.shape[0])
    valid_split = train_split+300
    data = data[shuffle_idx]
    label = label[shuffle_idx]

    train_data = data[:train_split]
    train_label = label[:train_split]

    valid_data = data[train_split:valid_split]
    valid_label = label[train_split:valid_split]

    test_data = data[valid_split:]
    test_label = label[valid_split:]

    np.savez(args.data_path+'/n_body_train.npz',
             data=train_data,
             label=train_label,
             n_particles=args.num_bodies,
             median=stat_median,
             max=stat_max,
             min=stat_min)

    np.savez(args.data_path+'/n_body_valid.npz',
             data=valid_data,
             label=valid_label,
             n_particles=args.num_bodies)

    test_traj = gen(args.num_bodies, args.steps, True)

    np.savez(args.data_path+'/n_body_test.npz',
             data=test_data,
             label=test_label,
             n_particles=args.num_bodies,
             first_frame=test_traj[0],
             test_traj=test_traj)
