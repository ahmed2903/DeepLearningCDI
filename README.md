# Deep Learning CDI

Code for  Deep Learning Phase Retrieval 

Expected shape of training data: 

    2 files: 1) reciprocal intensity file (n,1,x,y,z)
             2) real object file (n,2,x,y,z)

Training the neural network will produce a plot of the validation loss and the training loss at every epoch 

SGD is used for the backpropagation algorithm, other function can be used such as ADAM

Weight initilization: performed using built-in pytorch functions 
    for the batch normalization layer: all weights are set to 1 and biases to zero 
    for the Convultional layer: a Kaimin distribution is used 

    These initilization have proven to stabilize the network immensely 

Batch size: there is a memory limit of 32 batch size. Will have to edit the code to use multiple GPUs if we want to go higher than 32 

Learning rate schedular: 
    StepLR function which scales the LR by a factor Gamma every N steps (N is the learning rate step size which can be set)

Code has one class for training and can be run using the train_cnn.py file
        one class for prediction and can be run using the predict_obj.py file

