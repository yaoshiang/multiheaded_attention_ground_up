Ground up implementation of attention module / transformer model based on the ConvNext paper, Jay Alammar's Illustrated Transformer and a quick reference to the Attention Is All You Need paper to understand the final feed forward layer in each Multi Headed Attention model since ConvNext mentioned it resembled a 1x1 Point Conv - mixing information between heads rather than across tokens. 

Trains to 70% accuracy on Imagenette with very little hyperparameter or architecture tuning.

The MHA kernels are mainly implemented with tf.Variables and tf.einsum.

The Image Tokenizer is implemented as both a looping and a vectorized approach using tf.transpose and tf.reshape. 

No Keras layers for attention/transformers were used. 

Output from a few epochs:

```
/home/yaoshiang/miniconda3/envs/tf/lib/python3.8/site-packages/tensorflow_addons/utils/ensure_tf_install.py:53: UserWarning: Tensorflow Addons supports using Python ops for all Tensorflow versions above or equal to 2.7.0 and strictly below 2.10.0 (nightly versions are not supported). 
 The versions of TensorFlow you are currently using is 2.6.2 and is not supported. 
Some things might work, some things might not.
If you were to encounter a bug, do not file an issue.
If you want to make sure you're using a tested and supported configuration, either change the TensorFlow version or the TensorFlow Addons's version. 
You can find the compatibility matrix in TensorFlow Addon's readme:
https://github.com/tensorflow/addons
  warnings.warn(
Model: "model"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_1 (InputLayer)         [(None, 224, 224, 3)]     0         
_________________________________________________________________
image_tokenizer (ImageTokeni (None, 196, 64)           49344     
_________________________________________________________________
multi_headed_attention (Mult (None, 4, 196, 64)        83200     
_________________________________________________________________
multi_headed_attention_1 (Mu (None, 4, 196, 64)        83200     
_________________________________________________________________
multi_headed_attention_2 (Mu (None, 4, 196, 64)        83200     
_________________________________________________________________
multi_headed_attention_3 (Mu (None, 4, 196, 64)        83200     
_________________________________________________________________
tf.math.reduce_mean (TFOpLam (None, 4, 64)             0         
_________________________________________________________________
flatten (Flatten)            (None, 256)               0         
_________________________________________________________________
dense (Dense)                (None, 10)                2570      
=================================================================
Total params: 384,714
Trainable params: 384,714
Non-trainable params: 0
_________________________________________________________________
Epoch 1/100
252/252 [==============================] - 10s 22ms/step - loss: 1.9323 - acc: 0.3215 - val_loss: 1.6581 - val_acc: 0.4549
Epoch 2/100
252/252 [==============================] - 5s 21ms/step - loss: 1.4949 - acc: 0.5027 - val_loss: 1.3980 - val_acc: 0.5139
Epoch 3/100
252/252 [==============================] - 5s 21ms/step - loss: 1.3254 - acc: 0.5552 - val_loss: 1.2596 - val_acc: 0.5729
Epoch 4/100
252/252 [==============================] - 5s 21ms/step - loss: 1.2180 - acc: 0.5926 - val_loss: 1.2504 - val_acc: 0.5833
Epoch 5/100
252/252 [==============================] - 5s 21ms/step - loss: 1.1359 - acc: 0.6219 - val_loss: 1.0463 - val_acc: 0.6562
Epoch 6/100
252/252 [==============================] - 5s 21ms/step - loss: 1.0690 - acc: 0.6494 - val_loss: 1.0522 - val_acc: 0.6458
Epoch 7/100
252/252 [==============================] - 5s 21ms/step - loss: 1.0205 - acc: 0.6631 - val_loss: 1.0125 - val_acc: 0.6736
Epoch 8/100
252/252 [==============================] - 5s 21ms/step - loss: 0.9842 - acc: 0.6736 - val_loss: 1.0019 - val_acc: 0.6806
Epoch 9/100
252/252 [==============================] - 5s 21ms/step - loss: 0.9487 - acc: 0.6841 - val_loss: 0.9852 - val_acc: 0.6701
Epoch 10/100
252/252 [==============================] - 5s 21ms/step - loss: 0.9174 - acc: 0.6998 - val_loss: 1.1007 - val_acc: 0.6458
Epoch 11/100
252/252 [==============================] - 5s 21ms/step - loss: 0.8810 - acc: 0.7077 - val_loss: 0.9471 - val_acc: 0.7049
Epoch 12/100
252/252 [==============================] - 5s 21ms/step - loss: 0.8565 - acc: 0.7180 - val_loss: 0.9442 - val_acc: 0.6771
Epoch 13/100
252/252 [==============================] - 5s 21ms/step - loss: 0.8248 - acc: 0.7309 - val_loss: 0.8553 - val_acc: 0.7431
Epoch 14/100
252/252 [==============================] - 5s 21ms/step - loss: 0.8096 - acc: 0.7369 - val_loss: 0.8737 - val_acc: 0.7326
```


Copyright 2023 Yaoshiang Ho

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
