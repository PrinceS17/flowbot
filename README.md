
## FlowBot: A Learning-Based Co-bottleneck Flow Detector for Video Servers


### Prerequisites
Tested on Ubuntu 20.04, g++ 11.4.0, Python 3.8.10.

To install the python libraries, run
```bash
$ cd src
$ pip3 install -r requirements.txt
```

ns-3 configuration is not required, but if you want to run it manually, you can do
```bash
$ cd ns3/BBR_test/ns-3.27
$ CXXFLAGS="-Wall" ./waf configure
$ ./waf build
```

The dataset is stored in [flowbot-data](https://uofi.box.com/v/flowbot-data).
It provides the necessary data of the detection workflow.


### Introduction
The general workflow flow of FlowBot is:

```
config -> sim -> prep_train -> train            => Training set and model
             \-> prep_pred -> detect -> vis     => Test set and detection result
```

Meaning of the stages:
- config: generate ns-3 simulation configurations;
- sim: run ns-3 simulations and collect raw data;
- prep_train: preprocess the raw data and generate triplets for training;
- train: train the model using the triplets;
- prep_pred: preprocess the raw data and generate signals and ground truth for detection;
- detect: detect co-bottleneck flows and evaluate the accuracy;
- vis: visualize the detection results.


### Usage
TODO