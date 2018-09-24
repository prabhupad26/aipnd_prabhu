# Image Classifier application

This application is trained on 814 set of images of difference type of flowers using the torchvision's vgg16 model , checkout the sample output in the repo.

sample cmd to run :

python train.py flowers/ --gpu --arch vgg16 --hidden_units 512


python predict.py flowers/test/10/image_07090.jpg checkpoint --gpu --top_k 5

predict.py will generate:
1. Prediction and the top 5 prediction's probabilities. 
2. .png image with top_k classes and the image of the predicted image
