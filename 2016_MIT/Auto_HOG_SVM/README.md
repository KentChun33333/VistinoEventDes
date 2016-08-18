# Object Detect System
- Author : Kent (Jin-Chun Chiu)
- Goal : System for Object Detector 
- Feature : Using HOG + SVM for object detection in positive-vs-negtive model.

## Before running
1. Prepare your dataset ( Positive & Negative )
2. Config your parameters conf.json in conf_hub.

## Run the code
```shell
cd Auto_HOG_SVM/
python training_agent.py --conf conf_hub/conf_file.json 
```

## Configuration File
1. Configurations are in the `conf_hub/conf_xxx.json` JSON files. 
2. You can change it as per your need by pointing the right file_path in your environments.

