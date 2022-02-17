import keras as ks
import tensorflow as tf

#混同行列保存関数 kears
def ks_model_cfmat(model_path,valdat_path,val_start_pos,resoluion,byte_num,dat_size,dat_num,label_path,matrix_save_path):
    import seaborn as sns
    import matplotlib
    import dat_rb_func as drb
    import numpy as np
    import pandas as pd
    from sklearn.metrics import confusion_matrix
    dat=drb.sim_label_read(simdat_path=valdat_path,size=dat_size,num=dat_num,label_true=False,byte_num=byte_num)
    label=drb.sim_label_read(simdat_path=label_path,size=1,num=dat_num,label_true=True,byte_num=4)
    val=dat[val_start_pos:]
    val_label=label[val_start_pos:]

    val=val.astype('float32')
    val_label=val_label.astype('int64')
    val/=(resoluion-1)

    model=ks.models.load_model(model_path)

    test_predictions = model.predict(val)
    test_predictions=np.argmax(test_predictions, axis=1)

    _, ax = plt.subplots(figsize = (10, 10))

    cm = confusion_matrix(label_true = val_label, y_pred = test_predictions)
    sns.heatmap(cm,square=True,cmap='Blues',annot=True,fmt='d',ax=ax,vmax=50,vmin=0)
    plt.savefig(matrix_save_path)