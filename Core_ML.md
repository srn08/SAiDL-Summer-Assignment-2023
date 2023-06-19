## 1.Variations Of Softmax

### 1.1 First Trial - Basic CNN Architecture
This is the CNN architecture that I initially used:
```
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 conv2d (Conv2D)             (None, 30, 30, 30)        840       
                                                                 
 max_pooling2d (MaxPooling2D  (None, 15, 15, 30)       0         
 )                                                               
                                                                 
 conv2d_1 (Conv2D)           (None, 13, 13, 70)        18970     
                                                                 
 max_pooling2d_1 (MaxPooling  (None, 6, 6, 70)         0         
 2D)                                                             
                                                                 
 conv2d_2 (Conv2D)           (None, 4, 4, 160)         100960    
                                                                 
 flatten (Flatten)           (None, 2560)              0         
                                                                 
 dense (Dense)               (None, 64)                163904    
                                                                 
 dense_1 (Dense)             (None, 100)               6500      
                                                                 
=================================================================
Total params: 291,174
Trainable params: 291,174
Non-trainable params: 0
_________________________________________________________________
```

Now, let's train this basic model using both Softmax and Gumbel-Softmax and compare the results 

### 1.2 Training - Softmax vs Gumbel-Softmax
I have used Adam Optimizer and Categorical cross entropy as the loss function, we usually use SGD in such CNN based image classifiers, but Adam provides better accuracy than SGD in this case 
#### Training Time Softmax
All the following was done using Google Collab's T4 GPU
```
Epoch 1/10
1563/1563 [==============================] - 11s 6ms/step - loss: 3.9757 - accuracy: 0.0877 - val_loss: 3.5539 - val_accuracy: 0.1515
Epoch 2/10
1563/1563 [==============================] - 8s 5ms/step - loss: 3.2992 - accuracy: 0.2014 - val_loss: 3.1320 - val_accuracy: 0.2333
Epoch 3/10
1563/1563 [==============================] - 8s 5ms/step - loss: 2.9632 - accuracy: 0.2641 - val_loss: 2.9262 - val_accuracy: 0.2729
Epoch 4/10
1563/1563 [==============================] - 8s 5ms/step - loss: 2.7325 - accuracy: 0.3102 - val_loss: 2.7574 - val_accuracy: 0.3092
Epoch 5/10
1563/1563 [==============================] - 8s 5ms/step - loss: 2.5713 - accuracy: 0.3423 - val_loss: 2.6666 - val_accuracy: 0.3298
Epoch 6/10
1563/1563 [==============================] - 8s 5ms/step - loss: 2.4431 - accuracy: 0.3692 - val_loss: 2.6199 - val_accuracy: 0.3407
Epoch 7/10
1563/1563 [==============================] - 9s 6ms/step - loss: 2.3340 - accuracy: 0.3903 - val_loss: 2.5730 - val_accuracy: 0.3455
Epoch 8/10
1563/1563 [==============================] - 8s 5ms/step - loss: 2.2350 - accuracy: 0.4111 - val_loss: 2.5296 - val_accuracy: 0.3653
Epoch 9/10
1563/1563 [==============================] - 8s 5ms/step - loss: 2.1452 - accuracy: 0.4294 - val_loss: 2.5392 - val_accuracy: 0.3643
Epoch 10/10
1563/1563 [==============================] - 8s 5ms/step - loss: 2.0582 - accuracy: 0.4494 - val_loss: 2.5523 - val_accuracy: 0.3605

Accuracy: 0.3605
Precision: 0.3799
Recall: 0.3605
F1 Score: 0.3521
```
#### Training Same CNN with Gumbel Softmax Activation Function
```
Epoch 21/30
1563/1563 [==============================] - 8s 5ms/step - loss: 2.2017 - accuracy: 0.4598 - val_loss: 3.8224 - val_accuracy: 0.2992
Epoch 22/30
1563/1563 [==============================] - 9s 6ms/step - loss: 2.1664 - accuracy: 0.4689 - val_loss: 3.8476 - val_accuracy: 0.3088
Epoch 23/30
1563/1563 [==============================] - 8s 5ms/step - loss: 2.1183 - accuracy: 0.4738 - val_loss: 4.0555 - val_accuracy: 0.2986
Epoch 24/30
1563/1563 [==============================] - 8s 5ms/step - loss: 2.0564 - accuracy: 0.4892 - val_loss: 4.0123 - val_accuracy: 0.3086
Epoch 25/30
1563/1563 [==============================] - 9s 6ms/step - loss: 2.0151 - accuracy: 0.4992 - val_loss: 4.1067 - val_accuracy: 0.3083
Epoch 26/30
1563/1563 [==============================] - 8s 5ms/step - loss: nan - accuracy: 0.5057 - val_loss: 4.2307 - val_accuracy: 0.2952
Epoch 27/30
1563/1563 [==============================] - 9s 6ms/step - loss: 1.9253 - accuracy: 0.5169 - val_loss: 4.3594 - val_accuracy: 0.3013
Epoch 28/30
1563/1563 [==============================] - 9s 5ms/step - loss: 1.8774 - accuracy: 0.5243 - val_loss: 4.3232 - val_accuracy: 0.3020
Epoch 29/30
1563/1563 [==============================] - 8s 5ms/step - loss: 1.8327 - accuracy: 0.5326 - val_loss: 4.4539 - val_accuracy: 0.3071
Epoch 30/30
1563/1563 [==============================] - 9s 6ms/step - loss: 1.7892 - accuracy: 0.5427 - val_loss: nan - val_accuracy: 0.3024

Accuracy: 0.3024
Precision: 0.3044
Recall: 0.3023
F1 Score: 0.2981
```
