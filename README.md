# aipnd_prabhu
sample cmd to run :
python train.py flowers/ --gpu --arch vgg16


python predict.py flowers/test/10/image_07090.jpg checkpoint --gpu --top_k 5

predict.py will generate:
1. Prediction and the top 5 prediction's probabilities. 
2. .png image with top_k classes and the image of the predicted image


open issue:
1. hidden layers are hardcoded (couldn't find a better solution to make it dynamic)
