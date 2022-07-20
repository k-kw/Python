#import dat_rb_func as drb
import my_numpy_class as mnc
dataset_path = "kwmt/expefm_now/fm_55001_65000_bfshowslt0.4_N3_lsd.dat"
label_path = "kwmt/fm_label.dat"
width = 1600
savenum=30
#drb.simwave_ver2(30, 1600, dataset_path, label_path, 1, 15, 10, 15, 'position', 'data', './kwmt/simwave10fps2_3')
# data = drb.sim_label_read(dataset_path, 640, 1, False, 1)
# print(data)
labels = mnc.My_numpy(4, label_path)
labels.labelread(savenum)

sim = mnc.My_numpy(1, dataset_path)
sim.simread(savenum, width)
savepath = "kwmt/wave/fmwave"
sim.save_simwave(1, savenum, labels.data, 8, 4, 10, savepath)