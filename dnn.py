import numpy as np
import os
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mlxtend.plotting import plot_confusion_matrix

# config                           # activation function: relu / lk_relu / softmax
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
modelPath = 'model/784_512_512_20_10_lr5e-3_r1e-3_bs100_shf/ep240'
random = True
save_freq = 20


# math functions
def softmax(x):
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=0)


def relu(x):
    return np.maximum(0, x)

def lk_relu(x):
    return np.where(x > 0, x, x * 0.01)   

def d_tanh(x):
    return 1.0 - np.tanh(x)**2


def d_ce_softmax(y_hat, y):
    return y_hat - y


def d_relu(dJ, x):
    dx = np.array(dJ, copy=True)
    dx[x <= 0] = 0
    return dx


def dlk_relu(dJ, x):
    dx = np.array(dJ, copy=True)
    dx[x <= 0] *= -0.01
    return dx

# init model
def init_model(random):
    w = [None]*len(model)
    b = [None]*len(model)
    for layer_idx, layer_cont in enumerate(model):
        if random:
            w[layer_idx] = np.random.randn(layer_cont['output_d'], layer_cont['input_d']) * 0.1
            b[layer_idx] = np.random.randn(layer_cont['output_d'], bch_size) * 0.1
        else:
            w[layer_idx] = np.zeros((layer_cont['output_d'], layer_cont['input_d']))
            b[layer_idx] = np.zeros((layer_cont['output_d'], bch_size))

    return w, b


# propagation
def forward_prop(x, w, b):
    a = [None] * (len(model) +1)
    h = [None] * (len(model) +1)
    a[0] = 0
    h[0] = x.T
    for layer_idx, layer_cont in enumerate(model):
        a[layer_idx+1] = np.dot(w[layer_idx], h[layer_idx]) + b[layer_idx]

        if layer_cont['activation'] == 'relu':
            h[layer_idx+1] = relu(a[layer_idx+1])
        elif layer_cont['activation'] == 'lk_relu':
            h[layer_idx+1] = lk_relu(a[layer_idx+1])
        else:
            h[layer_idx + 1] = softmax(a[layer_idx+1])

    y_hat = h[layer_idx+1].T
    return a, h, y_hat


def backward_prop(y_hat, y, h, a, w, b):
    g = d_ce_softmax(y_hat, y.T)

    for layer_idx, layer_cont in reversed(list(enumerate(model))):
        m = h[layer_idx].shape[1]

        if layer_cont['activation'] == 'relu':
            g = d_relu(g, a[layer_idx+1])
        elif layer_cont['activation'] == 'lk_relu':
            g = dlk_relu(g, a[layer_idx+1])

        db = g / m
        dw = np.dot(g, h[layer_idx].T) / m
        g = np.dot(w[layer_idx].T, g)
        # update
        w[layer_idx] = (1-lr*reg)*w[layer_idx] - lr*dw
        b[layer_idx] = (1-lr*reg)*b[layer_idx] - lr*db

    return w, b


# loss function
def ce_loss(y_hat, y):
    m = y.shape[0]
    return -np.sum(y*np.log(y_hat+1e-9)) / m


# error rate
def error_r(y_hat, y):
    predict = np.zeros_like(y_hat)
    for i in range(y_hat.shape[0]):
        predict[i][np.argmax(y_hat[i])] = 1
    return 1-(predict == y).all(axis=1).mean()


def train(train_data, train_label, test_data, test_label):
    w, b = init_model(random)

    for i in range(epochs):
        train_error = 0
        test_error = 0
        loss = 0

        # shuffle training data
        state = np.random.get_state()
        np.random.shuffle(train_data)
        np.random.set_state(state)
        np.random.shuffle(train_label)

        # start training
        for j in range(12000 // bch_size):
            a, h, predict = forward_prop(train_data[j*bch_size:(j+1)*bch_size], w, b)
            loss = ce_loss(predict, train_label[j*bch_size:(j+1)*bch_size])
            print('epoch: %d step: [%d/%d] loss: %f'%(i, j, 12000 // bch_size, loss))
            logger.add_scalar('loss', loss, i*12000+j)
            train_error += error_r(predict, train_label[j*bch_size:(j+1)*bch_size])
            w, b = backward_prop(predict.T, train_label[j*bch_size:(j+1)*bch_size], h, a, w, b)

        logger.add_scalar('train_error', train_error / (12000 // bch_size), i)

        # start testing
        for j in range(5700 // bch_size):
            _, _, test_predict = forward_prop(test_data[j*bch_size:(j+1)*bch_size], w, b)
            test_error += error_r(test_predict, test_label[j*bch_size:(j+1)*bch_size])
        logger.add_scalar('test_error', test_error / (5700 // bch_size), i)

        # save model
        if i % save_freq == 0:
            os.mkdir('model/'+output_dir+'/ep'+str(i))
            for layer_idx, layer_cont in enumerate(model):
                np.save('model/'+output_dir+'/ep'+str(i)+'/w'+str(layer_idx), w[layer_idx])
                np.save('model/'+output_dir+'/ep'+str(i)+'/b'+str(layer_idx), b[layer_idx])

        print('epoch: %d train_error: %f test_error: %f' % (
        i, train_error / (12000 // bch_size), test_error / (5700 // bch_size))
                  )


def load_data():
    print('loading data')
    train_data = np.load('train.npz')
    test_data = np.load('test.npz')

    train_img = train_data['image'].reshape(12000, 784)
    label = train_data['label']
    train_label = np.zeros((12000, 10))
    for i in range(12000):
        train_label[i][int(label[i])] = 1

    test_img = test_data['image'].reshape(5768, 784)
    label = test_data['label']
    test_label = np.zeros((5768, 10))
    for i in range(5768):
        test_label[i][int(label[i])] = 1

    # set the image data range in [0, 1]
    train_img = train_img / 255.0
    test_img = test_img / 255.0

    return train_img, train_label, test_img, test_label


def load_model():
    print('Loading model: '+modelPath)
    w = [None]*len(model)
    b = [None]*len(model)

    for layer_idx, layer_cont in enumerate(model):
        w[layer_idx] = np.load(os.path.join(modelPath, 'w'+str(layer_idx)+'.npy'))
        b[layer_idx] = np.load(os.path.join(modelPath, 'b'+str(layer_idx)+'.npy'))
    return w, b


# visualization
def plot_scatter(predict, label):
    predict = np.array(predict)
    color = np.arange(10)
    num = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    fig = plt.figure()
    dis = fig.add_axes([0,0,1,1])
    clr = np.argmax(label, axis=1)
    s = dis.scatter(predict[:, :, 0], predict[:, :, 1], c=color[clr], cmap=cm.brg)
    
    plt.legend(handles=s.legend_elements()[0], labels=num, loc='lower right')
    #dis.legend()
    plt.savefig('300_epoch.jpg')


def get_2Df(train_data, train_label, test_data, test_label, w, b):
    predict_all = []

    # get 2-dim data
    for j in range(10):
        a = [None] * (len(model) +1)
        h = [None] * (len(model) +1)
        a[0] = 0
        h[0] = train_data[j*bch_size:(j+1)*bch_size].T
        for layer_idx, layer_cont in enumerate(model):
            a[layer_idx+1] = np.dot(w[layer_idx], h[layer_idx]) + b[layer_idx]

            if layer_idx == 2:
                predict_all.append(a[layer_idx+1].T)
                break

            if layer_cont['activation'] == 'relu':
                h[layer_idx+1] = relu(a[layer_idx+1])
            elif layer_cont['activation'] == 'lk_relu':
                h[layer_idx+1] = lk_relu(a[layer_idx+1])
            else:
                h[layer_idx + 1] = softmax(a[layer_idx+1])

    # plot scatter
    print('Start Plotting -----')
    plot_scatter(predict_all, train_label[:1000])

def get_confusion(test_data, test_label, w, b):
    confusion_array = np.zeros((10, 10), dtype=np.int16)
    for j in range(5700 // bch_size):
        _, _, test_predict = forward_prop(test_data[j*bch_size:(j+1)*bch_size], w, b)        
        predict = np.argmax(test_predict, axis=1)
        true = np.argmax(test_label[j*bch_size:(j+1)*bch_size], axis=1)
        for data in range(bch_size):
            confusion_array[true[data]][predict[data]] += 1

    # plot confusion matrix
    print('Confusion matrix -----')
    print(confusion_array)
    fig, ax = plot_confusion_matrix(conf_mat=confusion_array)
    fig.savefig('confusion_matrix.png')
        


if __name__ == '__main__':
    train_img, train_label, test_img, test_label = load_data()
    if istrain:
        train(train_img, train_label, test_img[:5700], test_label[:5700])
    else:
        w, b = load_model()
        #get_2Df(train_img, train_label, test_img[:5700], test_label[:5700], w, b)
        get_confusion(test_img[:5700], test_label[:5700], w, b)
