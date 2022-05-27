#import dat_rb_func as drb
import my_numpy_class as mnc
dataset_path = "kwmt/expem_now/m_25001_30000_slt0.6_N3_lsd.dat"
label_path = "kwmt/fm_label.dat"
width = 1600
#drb.simwave_ver2(30, 1600, dataset_path, label_path, 1, 15, 10, 15, 'position', 'data', './kwmt/simwave10fps2_3')
# data = drb.sim_label_read(dataset_path, 640, 1, False, 1)
# print(data)
labels = mnc.My_numpy(4, label_path)
labels.labelread(50)

sim = mnc.My_numpy(1, dataset_path)
sim.simread(50, width)
savepath = "kwmt/simwave10fps2_2"
sim.save_simwave(1, 30, labels.data, 8, 4, 10, savepath)