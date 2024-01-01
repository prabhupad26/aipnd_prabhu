# Flower type prediction with deep learning:

* Objective : Train an image classifier to recognize different species of flowers, with input image of size 224 x 224 .

* Solution : A neural network is created using a pretrained vgg16 model with its fully connected layer replaced with another set of fully connected layer with 102 outputs as we have to predict out of 102 different species of flowers.

* Results : The training results were achieved with the help of google colab GPU which took around 10 minutes to complete for processing a set of 1500 number of 224 x 224 sized images achieving an accuracy of 83 % while training and a testing accuracy of 86 %. This results were improved after considering data augmentation (rotating and resizing) the images during the preprocessing step.

CMD line application was created from the python noteboook for on demand training and inferencing the model, the training loop can save the model if the validation accuracy is improved.

> Sample command for running the cmd line application :
>
> For training the model :
> 
> `python train.py flowers/ --gpu --arch vgg16 --hidden_units 512`
> 
> Here the hidden units are the hidden layer input withing the fully connected layer
>
> For predicting with an input image :
>
> `python predict.py flowers/test/10/image_07090.jpg checkpoint --gpu --top_k 5`
> 
> predict.py will generate:
>  1. Prediction and the top 5 prediction's probabilities. 
>  2. .png image with top_k classes and the image of the predicted image
>
> ![Output sample](https://github.com/prabhupad26/aipnd_prabhu/blob/master/image_classifier.png)
