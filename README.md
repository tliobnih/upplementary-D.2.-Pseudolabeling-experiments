# Pseudolabeling experiments

Implementation of [Theoretical Analysis of Self-Training with Deep Networks on Unlabeled Data](https://arxiv.org/abs/2010.03622).

### Dependencies

```python
pip install -r requirements.txt
```

### Run code

Run sorce: mnist, target: svhn
```
python train.py --target_file svhn --seed 1
```

Run sorce: svhn, target: mnist 
```
python train.py --target_file mnist --seed 1
```

The accuracy will be saved to 'checkpoint'.

### Data Structure and Experimental Data
<img src="https://github.com/tliobnih/upplementary-D.2.-Pseudolabeling-experiments/assets/52643360/1a4f7597-9676-4c1b-b5b6-62c20c8c6777" width="50%" height="50%">

#### Experiment 1
Source: MNIST dataset with a selection of 10,000 data samples.  
Target: SVHN dataset with a selection of 10,000 training samples and 10,000 validation samples.
```
python train.py --target_file svhn --seed 1
```
The following data presents results obtained using different seeds. It can be observed that the experimental scores vary significantly across different seeds.  
<img src="https://github.com/tliobnih/upplementary-D.2.-Pseudolabeling-experiments/assets/52643360/6acc4031-2337-482e-8bbc-2062830d1d12" width="50%" height="50%">

Therefore, I conducted 50 experiments using seeds 1 to 50, and calculated the average of these 50 datas. The results are summarized in the following table:  
<img src="https://github.com/tliobnih/upplementary-D.2.-Pseudolabeling-experiments/assets/52643360/aa6edf85-70dc-43eb-9004-e75d3362ada3" width="30%" height="50%">  
The results of the fifty experiments are stored in "acc_svhn.csv", where each set of data can be replicated by simply changing the seed.

From the data, it appears that the accuracy does not exhibit the gradual increase as mentioned in the paper. I suspect that this may be due to the low scores in the first stage of the source dataset. Therefore, I conducted an additional experiment where I swapped the roles of the datasets. This is because using SVHN as the source dataset typically results in better training of the model.

#### Experiment 2
Source: SVHN dataset with a selection of 10,000 data samples.  
Target: MNIST dataset with a selection of 10,000 training samples and 10,000 validation samples.  
```
python train.py --target_file mnist --seed 1
```
<img src="https://github.com/tliobnih/upplementary-D.2.-Pseudolabeling-experiments/assets/52643360/5a525db2-a0cc-441f-bfeb-b047248275ad" width="30%" height="50%">  

Although the first-stage source results already achieved an accuracy of 62%, there is still no observed gradual increase as described in the paper. However, it is comforting to note that the third-stage PL+VAT approach yielded higher accuracy compared to the second-stage PL method. Of course, this is only an average observation, as not every instance of the PL+VAT method outperforms PL among the 50 seed-based datasets.  
Similarly, the data for these 50 experiments are stored in the dataset labeled "acc_mnist.csv", and each of them can reproduce the same results when rerun with the corresponding seed.  

#### Experiment 3
Source: MNIST dataset with a selection of 60,000 data samples.  
Target: SVHN dataset with a selection of 63257 training samples and 10,000 validation samples.  
```
python train.py --target_file svhn --seed 1  --num_mnist 60000  --num_svhn 63257  
```
<img src="https://github.com/tliobnih/upplementary-D.2.-Pseudolabeling-experiments/assets/52643360/21ddcc41-c7a8-4f78-954a-c3e4ca8196c3" width="40%" height="50%">  

Due to suspicions that the training dataset may not have been sufficient, all the available data from the dataset was included for training. However, the results remained unsatisfactory. Due to the large size of the dataset, only two sets of results were generated for this experiment, unlike Experiments 1 and 2, which involved running fifty times and averaging the results.  

#### Experiment 4
Source: SVHN dataset with a selection of 73257 data samples.  
Target: MNIST dataset with a selection of 50,000 training samples and 10,000 validation samples.    
```
python train.py --target_file svhn --seed 1  --num_mnist 50000  --num_svhn 73257
```
<img src="https://github.com/tliobnih/upplementary-D.2.-Pseudolabeling-experiments/assets/52643360/acffdcca-1a84-4d91-a932-1463f91a204d" width="50%" height="50%">  

Similar to Experiment 3, the results were still not significant.  

### Another Experimence
#### Experiment 5
<img src="https://github.com/tliobnih/upplementary-D.2.-Pseudolabeling-experiments/assets/52643360/37f09469-6191-4b67-a94e-5159ae3161cf" width="50%" height="50%">  
```
python train.py --target_file svhn --seed 1
```

Since this is an additional experiment that deviates from the paper, no extra parameters were set to control it. If you want to rerun this experiment, you will need to manually swap the comments of these two lines.  
<img src="https://github.com/tliobnih/upplementary-D.2.-Pseudolabeling-experiments/assets/52643360/fdb09b9d-8fe5-47c4-a8df-f40cf3e1f9bb" width="50%" height="50%">  


