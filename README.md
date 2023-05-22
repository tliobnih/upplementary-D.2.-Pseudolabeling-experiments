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

The following data presents results obtained using different seeds. It can be observed that the experimental scores vary significantly across different seeds.  
<img src="https://github.com/tliobnih/upplementary-D.2.-Pseudolabeling-experiments/assets/52643360/6acc4031-2337-482e-8bbc-2062830d1d12" width="50%" height="50%">

Therefore, I conducted 50 experiments using seeds 1 to 50, and calculated the average of these 50 datas. The results are summarized in the following table:  
<img src="https://github.com/tliobnih/upplementary-D.2.-Pseudolabeling-experiments/assets/52643360/aa6edf85-70dc-43eb-9004-e75d3362ada3" width="30%" height="50%">  
The results of the fifty experiments are stored in "acc_svhn.csv", where each set of data can be replicated by simply changing the seed.

From the data, it appears that the accuracy does not exhibit the gradual increase as mentioned in the paper. I suspect that this may be due to the low scores in the first stage of the source dataset. Therefore, I conducted an additional experiment where I swapped the roles of the datasets. This is because using SVHN as the source dataset typically results in better training of the model.

#### Experiment 2
Source: SVHN dataset with a selection of 10,000 data samples.  
Target: MNIST dataset with a selection of 10,000 training samples and 10,000 validation samples.  
<img src="https://github.com/tliobnih/upplementary-D.2.-Pseudolabeling-experiments/assets/52643360/5a525db2-a0cc-441f-bfeb-b047248275ad" width="30%" height="50%">  

Although the first-stage source results already achieved an accuracy of 62%, there is still no observed gradual increase as described in the paper. However, it is comforting to note that the third-stage PL+VAT approach yielded higher accuracy compared to the second-stage PL method. Of course, this is only an average observation, as not every instance of the PL+VAT method outperforms PL among the 50 seed-based datasets.  
Similarly, the data for these 50 experiments are stored in the dataset labeled "acc_mnist.csv", and each of them can reproduce the same results when rerun with the corresponding seed.  

#### Experiment 3
#### Experiment 4
"acc_svhn.csv"
