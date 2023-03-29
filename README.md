Ground up implementation of attention module / transformer model based on the ConvNext paper, Jay Alammar's Illustrated Transformer and a quick reference to the Attention Is All You Need paper to understand the final feed forward layer in each Multi Headed Attention model since ConvNext mentioned it resembled a 1x1 Point Conv - mixing information between heads rather than across tokens. 

Trains to 70% accuracy on Imagenette with very little hyperparameter or architecture tuning.

The MHA kernels are mainly implemented with tf.Variables and tf.einsum.

The Image Tokenizer is implemented as both a looping and a vectorized approach using tf.transpose and tf.reshape. 

No Keras layers for attention/transformers were used. 

Copyright 2023 Yaoshiang Ho

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
