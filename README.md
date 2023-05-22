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

### 資料架構與實驗數據

#### sorce: mnist 資料集選取10000筆資料
#### target: svhn 資料集選取10000(train)+10000(validation)筆資料

![image](https://github.com/tliobnih/upplementary-D.2.-Pseudolabeling-experiments/assets/52643360/1a4f7597-9676-4c1b-b5b6-62c20c8c6777)
