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
The following data presents results obtained using different seeds. It can be observed that the experimental accuracy vary significantly across different seeds.  
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

Since this is an additional experiment that deviates from the paper, no extra parameters were set to control it. If you want to rerun this experiment, you will need to manually swap the comments of these two lines.  
<img src="https://github.com/tliobnih/upplementary-D.2.-Pseudolabeling-experiments/assets/52643360/fdb09b9d-8fe5-47c4-a8df-f40cf3e1f9bb" width="50%" height="50%">  


From the table, it can be observed that in Experiment 5, which included the source data in addition to PL, the accuracy were significantly higher compared to Experiments 1 and 2 where only PL was used. Although the accuracy still did not show a gradual increase, it is expected that repeating Experiment 5 with different seeds for 50 iterations would yield better results compared to Experiments 1 and 2.  
### Parameter Settings
The parameter settings are as follows:  

PL+VAT:  
lambdav = 1. In the paper, the given parameters are 3, 10, and 30. Through experimentation, I found that the results with lambdav = 1 and lambdav = 3 are similar. Since the VAT loss decreases much faster than the other loss, I ultimately chose lambdav = 1 with the intention of not letting the VAT loss decrease too quickly. However, this hasn't been extensively tested. Due to time constraints, I didn't run 50 iterations with lambdav = 3, but it's worth trying.  

For the VAT implementation, I referred to https://github.com/sndnyang/vat_pytorch and made some modifications. As for the perturbation parameter, I chose to use its original settings. I did try to tweak it, but the effects were not significant. However, I believe this is an important parameter, and further tuning could potentially yield better results. With the current parameter configuration, the average perturbation is approximately 10^-6. Alternatively, you can consider using the "add_Gaussian_noise" function from https://github.com/CompVis/stable-diffusion/blob/main/ldm/modules/image_degradation/bsrgan.py, which might provide good results.  
<img src="https://github.com/tliobnih/upplementary-D.2.-Pseudolabeling-experiments/assets/52643360/6e8e31e5-75f9-4b17-baad-66788ddad779" width="50%" height="50%">    

Finally, for the model part, I used a simple ResNet50. The reason I didn't use a more complex model is that I believed that if the method described in the paper is effective, even if the initial source accuracy is low, both PL and PL+VAT accuracy would gradually increase. However, the experimental results showed that they did not improve over time.  

While the accuracy of PL and PL+VAT were not higher than the source accuracy, on average, VAT loss did show improvement in the PL accuracy compared to PL alone.
