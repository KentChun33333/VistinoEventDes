###################################
# Parameters

#----------------#
# DataSet Format #
#----------------#
# Main-folder
# -> Obj-A Folder ( with same resolution in the camera ? )
# -> Obj-B Folder
# -> Obj-N Folder


{
    #######
    # FOLDER PATH 
    #######
    "pos_raw_ph" : "/Users/kentchiu/MIT_Vedio/2D_DataSet/RHand",
    "pos_anno_ph":"",
    "neg_raw_ph" : "/Users/kentchiu/MIT_Vedio/2D_DataSet/BackGround",
    "hard_neg_ph" : "",
    "neg_feat_ph" : "/Users/kentchiu/feature_hub/neg/conf_001", # + conf-file-name/
    "pos_feat_ph" : "/Users/kentchiu/feature_hub/pos/conf_001", # + conf-file-name/
    "model_ph" : "model_hub/svm/conf_001b.model", # +conf-file.model/


    #######
    # DESCRIPTOR
    ######
    "orientations": 9,
    "pixels_per_cell": [8, 8],
    "cells_per_block": [3, 3],
    "visualize": false,
    "normalize": true,
    
    ######
    # Resolution 
    ######
    "raw_resolution":[],
    "train_resolution":[440,240], # ~detect_resolution
    "train_size":[40,40], # ~pos_obj size in train_resolution
    "sliding_size":[40,40], # must equal above, Detection slidinng window size

    #######
    # FEATURE EXTRACTION
    #######
    "percent_gt_images": 0.5, #how many persentage for traning from ground truth img
    "offset": 5, # ?
    "use_flip": true, #?
    "num_distraction_images": 500, # ?
    "num_distractions_per_image": 10, # ?

    #######
    # OBJECT DETECTOR, Feature No more than 1000
    #######
    "overlap_thresh": 0.3,
    "pyramid_scale": 1.5,
    "min_probability": 0.7,
    "step_size": 10,### n_feat.py,

    #######
    # LINEAR SVM
    #######
    "classifier_path": "output/cars/model.cpickle",
    "C": 0.01,

    #######
    # HARD NEGATIVE MINING
    #######
    "hn_num_distraction_images": 50,
    "hn_window_step": 4,
    "hn_pyramid_scale": 1.5,
    "hn_min_probability": 0.51
}