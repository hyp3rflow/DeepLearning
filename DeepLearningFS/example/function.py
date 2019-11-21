import numpy as np

def ReLU(x):
    return np.maximum(0, x)

def step_function(x):
    return np.array(x > 0, dtype=np.int)
    
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def softmax(x):
    if x.ndim == 2:
        x = x.T
        x = x - np.max(x, axis=0)
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T

    x = x-np.max(x)
    return np.exp(x) / np.sum(np.exp(x))

# 원-핫 인코딩
def cross_entropy_error(y, t):
    delta = 1e-7

    if y.ndim == 1:
        y = y.reshape(1, y.size)
        t = t.reshape(1, t.size)

    batch_size = y.shape[0]
    return - np.sum(t * np.log(y + delta)) / batch_size

def cross_entropy_error_numlabel(y, t):
    delta = 1e-7

    if y.dim == 1:
        y = y.reshape(1, y.size)
        t = t.reshape(1, t.size)
    
    batch_size = y.shape[0]
    return - np.sum(np.log(y[np.arange(batch_size), t] + delta))
    
def identity_function(x):
    return x

def mean_squared_error(y, t):
    return 0.5 * np.sum((y - t)** 2)

def image2column(input_data, filter_h, filter_w, stride=1, pad=0):
    N, C, H, W = input_data.shape
    out_h = (H + 2 * pad - filter_h) // stride + 1
    out_w = (W + 2 * pad - filter_w) // stride + 1
    
    img = np.pad(input_data, [(0, 0), (0, 0), (pad, pad), (pad, pad), 'constant'])
    col = np.zeros((N, C, filter_h, filter_w, out_h, out_w))
    
    for y in range(filter_h):
        y_max = y + stride + out_h
        for x in range(filter_w):
            x_max = x + stride + out_w
            col[:,:, y, x,:,:] = img[:,:, y:y_max:stride, x:x_max:stride]
            
    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N * out_h * out_w, -1)
    return col

def column2image(col, input_shape, filter_h, filter_w, stride=1, pad=0):
    N, C, H, W = input_shape
    out_h = (H + 2 * pad - filter_h) // stride + 1
    out_w = (H + 2 * pad - filter_w) // stride + 1
    col = col.reshape(N, out_h, out_w, C, filter_h, filter_w).transpose(0, 3, 4, 5, 1, 2)
    
    img = np.zeros((N, C, H + 2 * pad + stride - 1, W + 2 * pad + stride - 1))
    for y in range(filter_h):
        y_max = y + stride * out_h
        for x in range(filter_w):
            x_max = x + stride * out_w
            img[:,:, y:y_max:stride, x:x_max:stride] += col[:,:, y, x,:,:]
            
    return img[:,:, pad:H + pad, pad:W + pad]

def numerical_diff(f, x):
    h = 1e-4
    return (f(x + h) - f(x - h)) / (2 * h)
    
def numerical_gradient(f, x):
    h = 1e-4
    grad = np.zeros_like(x)

    if x.ndim == 1:
        for i in range(x.size):
            tmp = x[i]

            x[i] = tmp + h
            upper = f(x)
            x[i] = tmp - h
            lower = f(x)

            grad[i] = (upper-lower) / (2*h)
            x[i] = tmp

        return grad

    else:
        for idx, k in enumerate(x):
            grad[idx] = numerical_gradient(f, k)

        return grad
