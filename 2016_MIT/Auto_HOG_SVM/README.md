# Object Detect Sys
Author : Kent (Jin-Chun Chiu)
Goal : System for Object Detector


## Run the code
Ste
```shell
git clone https://github.com/bikz05/object-detector.git
cd object-detector/bin
test-object-detector
```

_The `test-object-detector` will download the [UIUC Image Database for Car Detection](https://cogcomp.cs.illinois.edu/Data/Car/) and train a classifier to detect cars in an image. The SVM model files will be stored in `data/models`, so that they can be resused later on._

### Configuration File

All the configurations are in the `data/config/config.cfg` configuration files. You can change it as per your need. Here is what the default configuration file looks like (which I have set for Car Detector)-


Step-by-Step

1. Prepare your dataset
2. config your parameters conf.json in conf_hub
3. use training_agent -conf conf.json