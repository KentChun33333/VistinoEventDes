# Object Detect System
- Author : Kent (Jin-Chun Chiu)
- Goal : System for Object Detector 
- Feature : Using HOG + SVM for object detection in positive-vs-negtive model.

## Before running
1. Prepare your dataset ( positive & negative )
2. config your parameters conf.json in conf_hub
3. use training_agent -conf conf.json

## Run the code

```shell
cd Auto_HOG_SVM/
python --conf conf_hub/conf_file.json 
```

### Configuration File

1. Configurations are in the `conf_hub/conf_xxx.json` JSON files. 
2. You can change it as per your need by pointing the right file_path in your environments.

