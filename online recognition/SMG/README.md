# Start kit for SMG dataset on online recognition

#### Citation

If you use our code or paper, please consider citing:
```
@article{chen2023smg,
  title={SMG: A Micro-Gesture Dataset Towards Spontaneous Body Gestures for Emotional Stress State Analysis},
  author={Chen, Haoyu and Shi, Henglin and Liu, Xin and Li, Xiaobai and Zhao, Guoying},
  journal={International Journal of Computer Vision},
  pages={1--21},
  year={2023},
  publisher={Springer}
}
```


## Dependencies
Keras 2.3.1
Tensorflow-gpu==1.15.0
cuda ==10.0 (cuda 10.1 is not compatible)
opencv
scikit-learn
matplotlib
openpyxl
pandas
xlrd


## Dataset preparation
Please download skeleton data from [this link](https://drive.google.com/file/d/1kzDFunbJz5ZFvdIBpNDTxGV3kecyWyPK/view?usp=share_link), and unzip it in the current code folder.

## Usage
We provide a demo code that has the complete development procedure, including data loading, model training and final evaluating.

The method is based on a DNN-HMM framework, please see the publication above for more details.

The usage of our code is easy, just run the code below.

```
python main.py
```

Your can change the hyperparameters according to your needs in the main.py file.


## Build up your own models

To access SMG samples, please see SMGaccessSample.py.

## License
MIT-2.0 License

