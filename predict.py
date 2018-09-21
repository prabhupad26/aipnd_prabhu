from PIL import Image
import matplotlib.pyplot as plt
from torchvision import datasets,transforms, models
import torch, json, argparse
import numpy as np
import argparse, train
import os
def main():
    input_arguments = input_argparse()
    im = Image.open(input_arguments.input_image_path)
    device = train.device_in_use(gpu_ind=input_arguments.gpu)
    label_to_name_json = cat_to_name_conv()
    model= train.build_model()
    model = train.load_checkpoint(checkpoint_loc = input_arguments.checkpoint_name+'.pth',model=model)
    probability, prediction = predict(image_path = im, model = model,topk=input_arguments.top_k , device = device)
    probability = probability.to('cpu')
    prediction = prediction.to('cpu')
    idx_to_class = {value: key for key, value in model.class_to_idx.items()}
    prediction.numpy()[0] = [idx_to_class[x] for x in prediction.numpy()[0]]
    top_classes = [label_to_name_json[str(x)] for x in  prediction.numpy()[0]]
    top_probabilities = probability.numpy()[0]
    print('predicted flower name :'+str(top_classes[0]))
    print('PROBABILITY'+' '+'PREDICTION')
    for probability, prediction in zip(top_probabilities ,top_classes):
        print(str(probability)+' : '+str(prediction))
    os.environ['QT_QPA_PLATFORM']='offscreen'
    show_result_image(image = im,probability= top_probabilities, top_classes = top_classes, data_dir = input_arguments.input_image_path)
def show_result_image(image, probability, top_classes, data_dir):
    fig, (ax1, ax2) = plt.subplots(figsize=(6,9), nrows=2)
    ax1.imshow(image)
    ax1.axis('off')
    ax1.set_title(top_classes[0])
    ax2.barh(top_classes,probability)
    ax2.set_xlim(0, 1.1)
    ax2.set_aspect(0.2)
    plt.savefig(data_dir.split('/')[2]+'_'+top_classes[0]+'.png')
def cat_to_name_conv():
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
    return cat_to_name
def process_image(image):
    normalize = transforms.Normalize(
       mean=[0.485, 0.456, 0.406],
       std=[0.229, 0.224, 0.225]
        )
    preprocess = transforms.Compose([
       transforms.Resize(256),
       transforms.CenterCrop(224),
       transforms.ToTensor(),
       normalize
    ])
    image_tensor = preprocess(image)
    return image_tensor
def imshow(image, ax=None, title=None):
    if ax is None:
        fig, ax = plt.subplots()
    image = image.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    image = np.clip(image, 0, 1)
    ax.imshow(image)
    return ax
def predict(image_path, model,device = 'gpu',topk=5):
    model.to(device)
    model.eval()
    with torch.no_grad():
        processed_image = process_image(image=image_path)
        processed_image = processed_image.to(device)
        outputs = model.forward(processed_image.view(1,3,224,224))
        probability,prediction = outputs.data.topk(k=topk)
        probability = torch.exp(probability)
    return probability,prediction
def input_argparse():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_image_path', type=str, default= 'flowers/test/10/image_07090.jpg',
                        help = 'set the directory where the image is present')
    parser.add_argument('checkpoint_name', type=str, default='checkpoint',
                        help='checkpoint name')
    parser.add_argument('--top_k', type = int, default=5,
                        help = 'set the number of top matching results')
    parser.add_argument('--category_names', type=str, default='cat_to_name.json',
                        help= 'label to name json filename')
    parser.add_argument('--gpu', action = 'store_true',
                        help='Enable cuda')
    return parser.parse_args()
if __name__ == '__main__':
    main()