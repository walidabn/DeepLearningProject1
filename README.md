# EE559-MiniProjet1
---
# How to use the library

Create 2 models:

* **Model1:**  (Image to label)
        - input:  1 channel image of 14x14 pixels
        - output: a vector of 10 values (representing the 10 labels)
* **Model2:** (label to difference)
        - input: 2 Vector of 10 labels
        - output: a single value, 0 if first label is bigger than second, 1 otherwise
        

The file `model_image_to_label.py` contains a variation of models that satisfies **Model1**

The file `model_labels_to_diff.py` contains a variation of models that satisfies **Model2**

To test the dimension of your networks:
```python
trainer.test_model_size(Model1(),Model2())
```

To combine these 2 models within one model taking 2 images and returning an binary class output you can use `model_full.py` as below
```python
# Construct a model that takes 2 images and return a single output
full_model = model_full.CombineNet(Model1(),Model2()) 

# Construct a model that takes 2 images and return the result and an auxiliary output that correspond to the labels of both images
full_model_aux = model_full.CombineWithLabels(Model1(),Model2())     
```

To train a model without auxiliary loss:
```python
results = train_model(model, # model to train
                      data["train_input"],  # train input
                      data["train_target_dif"].view(-1, 1), # train output
                      data["test_input"],# test input
                      data["test_target_dif"],# test output
                      epochs=<number>,eta=<number>,criterion=<loss_method>)
```

To train a model with auxiliary loss:
```python
results = train_model_auxiliary_loss(model, # model to train
                                     data["train_input"], # train input
                                     data["train_target_dif"].view(-1,1), # train output
                                     data["train_target_label"], # train auxiliary labels output
                                     data["test_input"], # test input
                                     data["test_target_dif"], # test output
                                     aux_weight =<number between 0 and 1>, # importance of auxiliary loss
                                     epochs=<number>,eta=<number>,#optionnal argument
                                     criterion=<loss_method>, # loss for output
                                     criterion_aux=<aux_loss_method> # loss for labels
                                    ) 
```
and finally you have the results where:
 * `result[0]` is an array containing the loss of your model for each epoch of the training
 * `result[1]` is an array containing the training error of your model for each epoch of the training
 * `result[2]` is an array containing the test error of your model for each epoch of the training

---
# Files
## test.py
Is a main method that train, evaluate and show the training and test errors of multiples different model

Unfortunately, no argument can be passed to the main through the console because dlc_practical_prologue.py is already using systems arguments. Hence the arguments have to be manually changed in the code itself 
 * batch_size 
 * epochs : number of epochs to train the models
 * n_data : number of data to load (default= 1000)
 * verbose: if true show the results of each epochs
 * plot: if true plots the training and testing error
 * repeat: the number of times we repeat the training for a model (ensure consistency of the result)
 * aux_loss: the weight accorded to the auxiliary loss compared to the final loss
 * eta: the learning rate

## trainer.py

Trainer possess a few methods that helps training and evaluating the models

```python
load_and_process_data(...)
test_model_size(...)
computeErrors(...)
train_model(...)
train_model_auxiliary_loss(...)
```
See the docstring for more info

## model_full.py

Contains multiples models class to predict if the first number is lower or equal to the second number

`CombineNet` takes a model that predict label of an image and a model that from the labels determine the the lowest number and use both of them to determine the lowest number directly from the images

`CombineWithLabels` does the same as previously but output the labels of both images as well

`NotSharedCombine`does the same as CombineNet but does not share the weights between the networks predicting labels for both images

`NotSharedCombineWithLabels` same as before with auxiliary output

## models_images_to_labels.py

Contains multiples Networks to determine the label of an image

`FullyConnected_I` uses 3 hidden layers of sizes [1000, 200, 100]

`LeNet5` is a modified version of LeNet5 architecture adapted for 14x14 images.  2 convolutions, one max pooling and 3 fully connected layer at the end

## models_labels_to_diff.py

Contains networks to compare 2 labels vector and return a binary class

`FullyConnected_II` a fully connected network with 2 hidden layer of sizes [100, 100]

`ArgMax`  is not really a model because you can't train it properly since the forward pass is deterministic. It assumes that the maximum value of the label is the correct one and use it to return if the first number is lower or equal to the second

## tools.py

 Contains few methods to vizualize images and create plots
