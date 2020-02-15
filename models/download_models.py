import torch
from os import path
from sys import version_info
from collections import OrderedDict
from torch.utils.model_zoo import load_url


# Download the VGG-19 model and fix the layer names
print("Downloading the VGG-19 model")
sd = load_url("https://doc-04-2g-docs.googleusercontent.com/docs/securesc/8g5vefd0ejd0m3k98a3073hltia11n4j/3eb4ecauoppipb87qgs9t7f0s9ctv25c/1581788700000/10186090967453425294/08452157988590945015/1QFE0T1v1jp9ZO7Sr2TDW_QDH_hIYSo2j?e=download&authuser=0")
map = {'classifier.1.weight':u'classifier.0.weight', 'classifier.1.bias':u'classifier.0.bias', 'classifier.4.weight':u'classifier.3.weight', 'classifier.4.bias':u'classifier.3.bias'}
sd = OrderedDict([(map[k] if k in map else k,v) for k,v in sd.items()])
torch.save(sd, path.join("models", "vgg19-d01eb7cb.pth"))

# Download the VGG-16 model and fix the layer names
print("Downloading the VGG-16 model")
sd = load_url("https://doc-0k-2g-docs.googleusercontent.com/docs/securesc/8g5vefd0ejd0m3k98a3073hltia11n4j/m6j41dq2ddkejel6hisljepgj6712c1p/1581788700000/10186090967453425294/08452157988590945015/1wcIMDvo2ZkbAZSkwYW32ON32qmIyLpJb?e=download&authuser=0")
map = {'classifier.1.weight':u'classifier.0.weight', 'classifier.1.bias':u'classifier.0.bias', 'classifier.4.weight':u'classifier.3.weight', 'classifier.4.bias':u'classifier.3.bias'}
sd = OrderedDict([(map[k] if k in map else k,v) for k,v in sd.items()])
torch.save(sd, path.join("models", "vgg16-00b39a1b.pth"))

# Download the NIN model
print("Downloading the NIN model")
if version_info[0] < 3:
    import urllib
    urllib.URLopener().retrieve("https://raw.githubusercontent.com/ProGamerGov/pytorch-nin/master/nin_imagenet.pth", path.join("models", "nin_imagenet.pth"))
else: 
    import urllib.request
    urllib.request.urlretrieve("https://raw.githubusercontent.com/ProGamerGov/pytorch-nin/master/nin_imagenet.pth", path.join("models", "nin_imagenet.pth"))

print("All models have been successfully downloaded")
