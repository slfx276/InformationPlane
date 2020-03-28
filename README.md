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
                                [-f FOLDER_NAME] [-re] [-show] [-cls]

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
  -re, --retrain        Retrain MNIST model and then store new representations
  -show, --showmine     show and save MINE training trend. (need GUI)
  -cls, --cleanfile     clean old data before creating new ones

```

### Usage 
```
python main.py -cls -re -show -bs 1024 -e 5 -bg 30 -amie 180 -mie 150
```
figures of MINE training would be saved in folder "mine".

### reference
- [how to plot multi-colored line with matplotlib](https://matplotlib.org/3.1.1/gallery/lines_bars_and_markers/multicolored_line.html)  
