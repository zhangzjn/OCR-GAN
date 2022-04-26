from options import Options
from lib.data.dataloader import load_data_FD_aug
from lib.models import load_model
import numpy as np
import prettytable as pt
##
def train(opt,class_name):
    data = load_data_FD_aug(opt, class_name)
    model = load_model(opt, data, class_name)
    auc = model.train()
    return auc
def main():
    """ Training
    """
    texture_classes = ["carpet", "grid", "leather", "tile", "wood"]
    object_classes = ["cable", "capsule", "hazelnut", "metal_nut", "pill", "screw", "toothbrush", "transistor", "zipper"]
    classes = texture_classes + object_classes
    texture_auc = []
    object_auc = []
    for i in classes:
        opt = Options().parse()
        auc = train(opt,i)
        if i in texture_classes:    
            texture_auc.append(auc)
        elif i in object_classes:
            object_auc.append(auc)
    all_auc = texture_auc+object_auc
    all_auc_avg = np.mean(np.array(all_auc))
    texture_auc_avg = np.mean(np.array(texture_auc))
    texture_auc.append(texture_auc_avg)
    texture_classes.append('avg')
    texture_classes.insert(0,'texture name')
    texture_auc.insert(0,'AUC')
    for i in range(4):
        texture_auc.append(' ')
        texture_classes.append(' ')
    
    object_auc_avg = np.mean(np.array(object_auc))
    object_auc.append(object_auc_avg)
    object_classes.append('avg')
    object_classes.insert(0,'object name')
    object_auc.insert(0,'AUC')
    import pdb
    pdb.set_trace()
    tb = pt.PrettyTable(header=False)
    tb.add_row(texture_classes)
    tb.add_row(texture_auc)
    tb.add_row(object_classes)
    tb.add_row(object_auc)
    print(tb)
if __name__ == '__main__':
    main()
