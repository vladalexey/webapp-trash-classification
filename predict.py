import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
from PIL import Image
# from trash_cnn_pytorch import train_model

def createModel(num_classes=6, w_drop=True):

    model_ft = models.resnext101_32x8d(pretrained=True)
    num_ftrs = model_ft.fc.in_features

    if not w_drop:
        model_ft.fc = nn.Linear(num_ftrs, num_classes)

    else:
        model_ft.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_ftrs, num_classes)
        )

    return model_ft

def get_transform():

    data_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    return data_transform
    
def setup(model_dir, model_class):

    # Check if saved model is loaded with CPU or GPU machine
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(model_dir, map_location=device)

    # Build model structure and optimizer 
    predictor = model_class(w_drop=False)
    opt = optim.SGD(predictor.parameters(), lr=0.001, momentum=0.9)

    # Load model weights and optimizer states
    predictor.load_state_dict(checkpoint['state_dict'])
    opt.load_state_dict(checkpoint['optimizer'])
    epoch = 0

    # Switch to evaluation mode
    predictor.eval()

    return predictor, opt, epoch

def predict(model, img, transform, epoch, classes=['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']):
    '''
        Do not change order of classes
    '''
    # Open with pillow
    img = Image.open(img)

    # Apply the same transform as training
    img_t = transform(img)
    batch_t = torch.unsqueeze(img_t, 0)

    # Forward the img thru the model
    res = model(batch_t)
    _, idx = torch.max(res, 1)

    preds = torch.nn.functional.softmax(res, dim=1)[0] * 100

    return classes[idx[0]], preds[idx[0]].item(), preds

# TODO: Finish converting train_model() to retrain purpose
# def retrain(model, opt, imgs, transform, start_epoch=0, classes=['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']):
#     '''
#         Continue training on minibatch of new observations
#     '''
#     if len(imgs) < 40:

#         print("Not enough training data")
#         return
    
#     criterion = nn.CrossEntropyLoss()

#     # Decay LR by a factor of 0.1 every 7 epochs
#     scheduler = lr_scheduler.StepLR(opt, step_size=5, gamma=0.1)

#     new_model_ft, best_acc, loss = train_model(model, criterion, opt, scheduler, start_epoch, num_epochs=5)

#     checkpoint = {
#         'epoch': start_epoch + 5,
#         'model': createModel(),
#         'model_state_dict': new_model_ft.state_dict(),
#         'optimizer_state_dict': opt.state_dict()
#     }

#     torch.save(checkpoint, 'garbage-classification/models_resnext101_32x8d_acc: {:g} loss: {:g}'.format(best_acc, loss))

if __name__ == "__main__":

    sample_inp = 'sample.jpg'
    model_dir = './models/models_resnext101_32x8d_acc_ 0.951807 loss_ 0.18151'

    data_transform = get_transform()

    predictor, opt, epoch = setup(model_dir, createModel)
    pred, conf, preds = predict(predictor, sample_inp, data_transform, epoch)

    print("Prediction: {} at {:g} confidence \nConf list: {}".format(pred, conf, preds))