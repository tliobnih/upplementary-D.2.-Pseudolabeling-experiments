# Pseudolabeling experiments

Implementation of [Theoretical Analysis of Self-Training with Deep Networks on Unlabeled Data](https://arxiv.org/abs/2010.03622).

### Dependencies

```python
pip install -r requirements.txt
```

### Run code

Run sorce: mnist target: svhn
```
python train.py --target_file svhn --seed 1
```

Run sorce: svhn target: mnist 
```
python train.py --target_file mnist --seed 1
```

The accuracy will be saved to 'checkpoint'.

### Data Structure and Experimental Data
Source: MNIST dataset with a selection of 10,000 data samples.  
Target: SVHN dataset with a selection of 10,000 training samples and 10,000 validation samples.

<img src="https://github.com/tliobnih/upplementary-D.2.-Pseudolabeling-experiments/assets/52643360/1a4f7597-9676-4c1b-b5b6-62c20c8c6777" width="50%" height="50%">

The following data presents results obtained using different seeds. It can be observed that the experimental scores vary significantly across different seeds.  
<img src="https://github.com/tliobnih/upplementary-D.2.-Pseudolabeling-experiments/assets/52643360/6acc4031-2337-482e-8bbc-2062830d1d12" width="50%" height="50%">


Therefore, I conducted 50 experiments using seeds 1 to 50, and calculated the average of these 50 datas. The results are summarized in the following table:  
<img src="https://github.com/tliobnih/upplementary-D.2.-Pseudolabeling-experiments/assets/52643360/aa6edf85-70dc-43eb-9004-e75d3362ada3" width="30%" height="50%">

