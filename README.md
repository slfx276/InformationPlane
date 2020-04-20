# InformationPlane
Calculating mutual information while training MNIST model by mutual information estimator  

### Papers
- [Opening the black box of Deep Neural Networks via Information](https://arxiv.org/pdf/1703.00810.pdf)
- [Mutual Information Neural Estimation](https://arxiv.org/pdf/1801.04062.pdf)
- [THE EFFECTIVENESS OF LAYER-BY-LAYER TRAINING USING THE INFORMATION BOTTLENECK PRINCIPLE](https://openreview.net/pdf?id=r1Nb5i05tX)

### Environment
- pytorch-gpu
- MNIST dataset (will be downloaded automatically.)

### Arguments
```
usage: Create inputs of main.py [-h] [-bs BATCH_SIZE] [-e MNIST_EPOCH]
                                [-var NOISE_VAR] [-mie MINE_EPOCH]
                                [-amie AAMINE_EPOCH] [-bg BATCH_GROUP]
                                [-f FOLDER_NAME] [-opt MNIST_OPT]
                                [-lr MNIST_LR] [-re] [-show] [-cls]
                                [-m MODEL_TYPE]

optional arguments:
  -h, --help            show this help message and exit
  -bs BATCH_SIZE, --batchsize BATCH_SIZE
                        set training batch size of MNIST model
  -e MNIST_EPOCH, --mnistepoch MNIST_EPOCH
                        set training epochs of MNIST model
  -var NOISE_VAR, --noisevariance NOISE_VAR
                        noise variance in noisy representation while using
                        MINE
  -mie MINE_EPOCH, --mineepoch MINE_EPOCH
                        training epochs of MINE model while estimating mutual
                        information
  -amie AAMINE_EPOCH, --amineepoch AAMINE_EPOCH
                        how many batch do you want to combined into a group in
                        order to calculate MI
  -bg BATCH_GROUP, --bgroup BATCH_GROUP
                        how many batch do you want to combined into a group in
                        order to calculate MI
  -f FOLDER_NAME, --folder FOLDER_NAME
                        the name of folder which you create for saving MINE
                        training trend.
  -opt MNIST_OPT, --optimizer MNIST_OPT
                        the optimizer used to train MNIST model.
  -lr MNIST_LR, --lr MNIST_LR
                        initial learning rate used to train MNIST model.
  -re, --retrain        Retrain MNIST model and then store new representations
  -show, --showmine     show and save MINE training trend. (need GUI)
  -cls, --cleanfile     clean old data before creating new ones
  -m MODEL_TYPE, --nntype MODEL_TYPE
                        NN model type could be mlp or cnn.



```

### Usage Example
```
python main.py -cls -re -show -m cnn -bs 1024 -bg 59 -e 10 -mie 200 -amie 200 -opt adam -var 0.03 -lr 0.01 -f bs1024bg59e10mie200amie200adamvar003lr001
```
Then figures of MINE training process would be saved in folder "bs1024bg59e10mie200amie200adamvar003lr001".  
and information plane would be saved in current path.

when using main.py, you should at least enter arguments below
```
CUDA_VISIBLE_DEVICES=0 time python main.py -cls -re -show -m cnn
```
when using other GPU to assist
```
CUDA_VISIBLE_DEVICES=0 time python main.py
```
when using plot.py
```
python plot.py
```
use utils.py to check current MI calculation result
```
python utils.py
```
![image](https://github.com/slfx276/InformationPlane/blob/includeCNN/ip_bs4096_e10_var2.0_bg59_adam_lr0.001_mie250_amie1400_typemlptanh500_256_10__testtanh_NOISE2_2_005.png)



### reference
- [how to plot multi-colored line with matplotlib](https://matplotlib.org/3.1.1/gallery/lines_bars_and_markers/multicolored_line.html)  
