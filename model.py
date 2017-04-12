#!/usr/bin/python3

import argparse
import cv2
import numpy as np
import pandas as pd

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Convolution2D

from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras_diagram import ascii


nvidia_input_size = (200,66)


def pre_process(img):
    '''Resize image it to match desired neural net input,
    Normalise between -1 and 1'''
    img = cv2.resize(img, nvidia_input_size, interpolation=cv2.INTER_AREA)
    return img/127.5 - 1.0

def flip(img, steer):
    '''Flip image vertically'''
    return cv2.flip(img, 1), -steer

def translate_crop(img, steer, y_trans_max=2, x_trans_max=20, y_crop=(60,25), random=True, dx_dsteer=-.001):
    '''Translate image horizontally by a random number of pixel vertically and horizontally'''
    if random:
        dx = int(x_trans_max*(2*np.random.uniform()-1))
        dy = int(y_trans_max*(2*np.random.uniform()-1))
    else:
        dx = 0
        dy = 0
        
    rows, cols, d = img.shape
    
    img = img[y_crop[0]:rows-y_crop[1], x_trans_max+dx:cols-x_trans_max+dx, :]
    steer += dx_dsteer * dx
    
    return img, steer

def brightness(img):
    '''Modify image brightness (between 80 and 120% of original image'''
    random_bright = .8 + .4*np.random.uniform()
    
    img = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    img[:,:,2] = img[:,:,2]*random_bright
    img = cv2.cvtColor(img,cv2.COLOR_HSV2RGB)
    
    return img

def shear(img,steer,shear_range=50, dshear_dsteer=.005):
    '''Shear the image and adapt the steering angle accordingly'''
    rows,cols,d = img.shape
    
    # shear distance
    dshear = int(shear_range*(2*np.random.uniform()-1))
    new_center = [cols/2+dshear,rows/2]
    
    # source -> dest
    source = np.float32([[0,rows],[cols,rows],[cols/2,rows/2]])
    dest = np.float32([[0,rows],[cols,rows],new_center])
    
    dsteering = dshear * dshear_dsteer
    
    M = cv2.getAffineTransform(source,dest)
    img = cv2.warpAffine(img,M,(cols,rows),borderMode=1)
    steer +=dsteering
    
    return img,steer

def augment_from_row(row, use_side_cam=False, offset=0.12):
    '''Read one of the three images from the given row, modify it with flip/translation/brightness change'''
    
    # select image and flip or not
    flip_it = np.random.choice([False, True])
    
    
    d_steering = 0
    if use_side_cam:
        side = np.random.choice(['center', 'left', 'right'])
        if side == 'left':
            d_steering += offset
        elif side == 'right':
            d_steering -= offset
    else:
        side = 'center'
    
    
    # read the image
    img = cv2.imread(row[side][0])
    steer = row['steering'][0] + d_steering
    
    if flip_it:
        img, steer = flip(img, steer)
        
    img, steer = shear(img, steer)

    img, steer = translate_crop(img, steer)
    
    img = brightness(img)
    
    return img, steer


def batch_generator(df, batch_size, augment=True):
    '''Generate augmented image dataset'''
    X = np.zeros((batch_size, *nvidia_input_size, 3))
    y = np.zeros((batch_size))
    while True:
        for i in range(batch_size):
            idx = np.random.randint(len(df))
            if augment:
                row = df.iloc[[idx]].reset_index()
                img, steer = augment_from_row(row)
            else:   # for validation
                row = df.iloc[[idx]].reset_index()
                steer = row['steering'][0]
                if abs(steer) < .001:   # reduce the number of sample with a 0 steering angle
                    row = df.iloc[[idx]].reset_index()
                    steer = row['steering'][0]
                img = cv2.cvtColor(cv2.imread(row['center'][0]), cv2.COLOR_BGR2RGB)
                img,dummy = translate_crop(img, 0, random=False)
            X[i] = np.transpose(pre_process(img), axes=[1,0,2])
            y[i] = steer
        yield X,y


def nvidia_model():
    '''Generate Keras model for the network in the nvidia paper'''
    model = Sequential()
    
    model.add(Convolution2D(24, 5, input_shape=(*nvidia_input_size, 3), padding='valid', strides=(2, 2)))
    model.add(Activation('relu'))
    model.add(Dropout(.2))
    
    model.add(Convolution2D(36, 5, padding='valid', strides=(2, 2)))
    model.add(Activation('relu'))
    
    model.add(Convolution2D(48, 5, padding='valid', strides=(2, 2)))
    model.add(Activation('relu'))
    
    model.add(Convolution2D(64, 3, padding='valid'))
    model.add(Activation('relu'))
    
    model.add(Convolution2D(64, 3, padding='valid'))
    model.add(Activation('relu'))
    
    model.add(Flatten())
    
    model.add(Dense(1024, init='normal'))
    model.add(Activation('relu'))
    model.add(Dropout(.3))
    
    model.add(Dense(100))
    model.add(Activation('relu'))
    model.add(Dropout(.3))
    
    model.add(Dense(50))
    model.add(Activation('relu'))
    
    model.add(Dense(10))
    model.add(Activation('relu'))
    
    model.add(Dense(1))
    
    return model
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SDCND Behavioural Cloning -- Training')
    parser.add_argument('--data', dest='data', type=str, default='./data/',
                        help='Path to input data')
    parser.add_argument('--output', dest='output', type=str, default='./model.h5',
                        help='Path where to save the trained model')
    parser.add_argument('--epochs', dest='epochs', type=int, default=100,
                        help='Number of epochs')
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=256,
                        help='Batch size')
    parser.add_argument('--batch_per_epoch', dest='batch_per_epoch', type=int, default=20,
                        help='Batch per epoch')
    args = parser.parse_args()
    
    
    # read the dataframe
    df = pd.read_csv(args.data + '/driving_log.csv')
    df['left'] = args.data + '/' + df['left'].str.lstrip()
    df['right'] = args.data + '/' + df['right'].str.lstrip()
    df['center'] = args.data + '/' + df['center'].str.lstrip()
    
    # create the neural network
    model = nvidia_model()
    print(ascii(model))

    model.compile(optimizer='adam', loss='mse')
    
    es = EarlyStopping(monitor='val_loss', patience=3)
    cp = ModelCheckpoint('../weights.{epoch:02d}-{val_loss:.6f}.hdf5', monitor='val_loss')


    
    model.fit_generator(batch_generator(df, batch_size=args.batch_size), args.batch_per_epoch, args.epochs, callbacks=[es, cp], validation_data=batch_generator(df, batch_size=args.batch_size, augment=False), validation_steps=2)
    
    model.save(args.output)



