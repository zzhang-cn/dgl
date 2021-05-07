from __future__ import print_function, division
import os
import sys
import tqdm
import numpy as np
from multiprocessing import Pool, TimeoutError

from dataset.preprocess_mesh import preprocess as preprocess_mesh_obj

def get_all_obj_file_in_folder(data_dir, all_list_path):
    class_dirs = [os.path.join(data_dir, cdir) for cdir in os.listdir(data_dir) if 
                 os.path.isdir(os.path.join(data_dir, cdir))]
    print (class_dirs)
    with open(all_list_path, 'w') as f:
        for class_dir in class_dirs:
            file_dirs = [os.path.join(class_dir, cdir) for cdir in os.listdir(class_dir) if 
                         os.path.isdir(os.path.join(class_dir, cdir))]
            for f_dir in file_dirs:
                obj_file  = os.path.join(os.path.join(f_dir, 'models'), 'model_normalized.obj')
                if os.path.isfile(obj_file):
                    f.write(obj_file+'\n')

def train_test_split(all_list_path):
    train_part = 0.925
    val_part = 0.025
    test_part = 0.05

    with open(all_list_path, 'r') as f:
        all_list = [line.strip() for line in f.readlines()]
    num_files = len(all_list)
    num_files_train = int(num_files * train_part)
    num_files_val = int(num_files * val_part)
    arr = np.arange(num_files)
    np.random.shuffle(all_list)
    train_list = all_list[:num_files_train]
    val_list = all_list[num_files_train:num_files_train+num_files_val]
    test_list = all_list[num_files_train+num_files_val:]
    with open(all_list_path+'.train', 'w') as f:
        for t_f in train_list:
            f.write(t_f+'\n')
    with open(all_list_path+'.val', 'w') as f:
        for t_f in val_list:
            f.write(t_f+'\n')
    with open(all_list_path+'.test', 'w') as f:
        for t_f in test_list:
            f.write(t_f+'\n')



def check_3d_obj_len(obj_file):
    node_num = 800
    face_num = 2400
    bin_num = 256
    try:
        verts, faces = preprocess_mesh_obj(obj_file.strip(), bin_num)
    except:
        return None
    if verts.shape[0] > node_num or faces.shape[0]*3 > face_num:
        return None
    return obj_file.strip()


def filter_dataset(ori_list_path):
    with open(ori_list_path, 'r') as f:
        all_list = [line.strip() for line in f.readlines()]
    pool = Pool(processes=64)
    filtered_list = []
    for res in tqdm.tqdm(pool.imap_unordered(check_3d_obj_len, all_list), total=len(all_list)):
        if res:
            filtered_list.append(res)

    with open(ori_list_path[:-4]+'_filtered.txt', 'w') as f:
        for f_file in filtered_list:
            f.write(f_file+'\n')
 

if __name__ == '__main__':
    np.random.seed(1111)
    data_dir = '/home/ubuntu/data/ShapeNetCore.v2/'
    all_list_path = os.path.join(data_dir, sys.argv[1])
    print (all_list_path)
    #get_all_obj_file_in_folder(data_dir, all_list_path)
    train_test_split(all_list_path)
    #filter_dataset(all_list_path)
