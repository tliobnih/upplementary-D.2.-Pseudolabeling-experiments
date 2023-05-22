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
#### Source: MNIST dataset with a selection of 10,000 data samples.
#### Target: SVHN dataset with a selection of 10,000 training samples and 10,000 validation samples.

![image](https://github.com/tliobnih/upplementary-D.2.-Pseudolabeling-experiments/assets/52643360/1a4f7597-9676-4c1b-b5b6-62c20c8c6777)
The following data presents results obtained using different seeds. It can be observed that the experimental scores vary significantly across different seeds.
![image](https://github.com/tliobnih/upplementary-D.2.-Pseudolabeling-experiments/assets/52643360/6acc4031-2337-482e-8bbc-2062830d1d12)
