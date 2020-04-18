# DNN-for-classification
implement a DNN with numpy

## Setup
1. 創建一個新環境
```
python3 -m venv env_name
```
2. activate environment
```
source env_name/bin/activate
```
3. 安裝requirement.txt中的套件
```
pip3 install -r requirements.txt
```


## Download Data
1. 使用TibetanMNIST: 藏文手寫數字數據集 (https://github.com/bat67/TibetanMNIST)
2. 下載 Dataset > TibetanMNIST.npz (28*28, 1 channel)
3. split 12000 img for training data, 5768 img for testing data, name it as 'train.npz' / 'test.npz' 



## Training
1. 修改dnn.py中的config
```python
# CONFIG              # activation function: relu / lk_relu / softmax
model = [
    {'input_d': 784, 'output_d': 512, 'activation': 'lk_relu'},
    {'input_d': 512, 'output_d': 512, 'activation': 'lk_relu'},
    {'input_d': 512, 'output_d': 20, 'activation': 'lk_relu'},
    {'input_d': 20, 'output_d': 10, 'activation': 'softmax'}
]

output_dir = '784_512_512_20_10_lr5e-3_r1e-3_bs100_shf_lkrelu'
if not os.path.isdir('model/'+output_dir):
    os.mkdir('model/'+output_dir)
logger = SummaryWriter('log/'+output_dir)

epochs = 250
bch_size = 100
lr = 0.005
reg = 1e-3
istrain = True
modelPath = ''
random = True
save_freq = 20

```
### configuration
- **model** - model structure.
- **output_dir** - model name.
- **epochs** - epoch number.
- **bch_size** - batch size.
- **lr** - learning rate.
- **reg** - regularization term
- **istrain** - if the model do train or test
- **modelPath** - model path for testing, work if istrain=False
- **random** - if the initial weight random
- **save_freq** - save modal every save_freq epochs


2. run dnn.py
```
python3 dnn.py
```

### tensorboardX
可以使用tensorboard觀察loss及error rate變化
```
tensorboard --logdir log
```

## Testing
1. 修改istrain及modelPath
```python
istrain = False
modelPath = 'model/784_512_512_20_10_lr5e-3_r1e-3_bs100_shf/ep240'
```
2. run dnn.py, save a confusion matrix img for test data
```
python3 dnn.py
```

3. 如果model output的前一層node為2，可以uncomment get2Df function對2維feature做visualization
```python
if __name__ == '__main__':
    train_img, train_label, test_img, test_label = load_data()
    if istrain:
        train(train_img, train_label, test_img[:5700], test_label[:5700])
    else:
        w, b = load_model()
        get_2Df(train_img, train_label, test_img[:5700], test_label[:5700], w, b)
        get_confusion(test_img[:5700], test_label[:5700], w, b)
```
