import dat_rb_func as drb
dataset_path = "kwmt/m_1_5000_fps10_lsd.dat"
label_path = "kwmt/mnist_label.dat"
width = 1600
drb.simwave_ver2(40, 1600, dataset_path, label_path, 1, 15, 10, 15, 'position', 'data', './kwmt/simwave_')
# data = drb.sim_label_read(dataset_path, 640, 1, False, 1)
# print(data)