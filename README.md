**This repository is a little bit messy, it hasn't been refactorized yet...**

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
CUDA_VISIBLE_DEVICES=[GPU number] time python main.py -cls -re -show -f sigmoid100epochs -m mlpsigmoid -e 100 -mie 500 -amie 1500
```
Then figures of AA-MINE & MINE training process would be saved in folder "sigmoid100epochs".      
All representations would be saved in directory "repre" until the next excution.

if you want to speed up calculating, you could use command below with different GPU (after finishing MNIST model training).
```
CUDA_VISIBLE_DEVICES=[Other GPUs] python main.py     # Do not use arguments especially -re & -cls.
```
Finally, use plots.py to gather all MI values and then visualize them on information plane.
```
python plots.py
```
  

![image](https://github.com/slfx276/InformationPlane/blob/master/ip_bs4096_e100_var2_bg59_adam_lr0.001_mie500_amie1500_typemlpsigmoid500_256_10__sigmoid100epochs.png)  
  
(Optional)  
You could use utils.py to check the rate of progress on calculating MI.  
```
python utils.py
```

### reference
- [how to plot multi-colored line with matplotlib](https://matplotlib.org/3.1.1/gallery/lines_bars_and_markers/multicolored_line.html)  
- [Mine_pytorch](https://github.com/MasanoriYamada/Mine_pytorch/blob/master/mine.ipynb)
