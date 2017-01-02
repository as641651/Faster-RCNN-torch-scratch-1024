import caffe
import PyTorch
from PyTorchAug import nn
import PyTorchAug


def transferW(b,a):
  print b.size()
  print a.shape
  for i in range(b.size()[0]):
     for j in range(b.size()[1]):
        b[i][j] = a[i][j]

def transferB(b,a):
  print b.size()
  print a.shape
  for i in range(b.size()[0]):
      b[i] = a[i]

tnet = {}

#net = caffe.Net('test.prototxt','/work/cv3/sankaran/faster-rcnn/data/imagenet_models/VGG_CNN_M_1024.v2.caffemodel',caffe.TEST)
net = caffe.Net('test.prototxt','/home/as641651/user/faster-rcnn/data/imagenet_models/VGG_CNN_M_1024.v2.caffemodel',caffe.TEST)
print net.params['fc6'][0].data.shape
print net.params['fc6'][1].data.shape

tnet["conv1"] = nn.SpatialConvolutionMM(3,96,7,7,2,2,0,0)
transferW(tnet["conv1"].weight, net.params['conv1'][0].data.reshape(96,147))
transferB(tnet["conv1"].bias, net.params['conv1'][1].data)
tnet["relu1"] = nn.ReLU()
tnet["norm1"] = nn.SpatialCrossMapLRN(5,0.0005,0.75,2)
tnet["pool1"] = nn.SpatialMaxPooling(3,3,2,2,0,0)

tnet["conv2"] = nn.SpatialConvolutionMM(96,256,5,5,2,2,1,1)
transferW(tnet["conv2"].weight, net.params['conv2'][0].data.reshape(256,2400))
transferB(tnet["conv2"].bias, net.params['conv2'][1].data)
tnet["relu2"] = nn.ReLU()
tnet["norm2"] = nn.SpatialCrossMapLRN(5,0.0005,0.75,2)
tnet["pool2"] = nn.SpatialMaxPooling(3,3,2,2,0,0)

tnet["conv3"]= nn.SpatialConvolutionMM(256,512,3,3,1,1,1,1)
transferW(tnet["conv3"].weight, net.params['conv3'][0].data.reshape(512,2304))
transferB(tnet["conv3"].bias, net.params['conv3'][1].data)
tnet["relu3"] = nn.ReLU()

tnet["conv4"]= nn.SpatialConvolutionMM(512,512,3,3,1,1,1,1)
transferW(tnet["conv4"].weight, net.params['conv4'][0].data.reshape(512,4608))
transferB(tnet["conv4"].bias, net.params['conv4'][1].data)
tnet["relu4"] = nn.ReLU()

tnet["conv5"]= nn.SpatialConvolutionMM(512,512,3,3,1,1,1,1)
transferW(tnet["conv5"].weight, net.params['conv5'][0].data.reshape(512,4608))
transferB(tnet["conv5"].bias, net.params['conv5'][1].data)
tnet["relu5"] = nn.ReLU()


tnet["rpn_conv/3x3"]= nn.SpatialConvolutionMM(512,256,3,3,1,1,1,1)
tnet["rpn_relu/3x3"] = nn.ReLU()

tnet["rpn_cls_score"]= nn.SpatialConvolutionMM(256,18,1,1,1,1,0,0)

tnet["rpn_bbox_pred"]= nn.SpatialConvolutionMM(256,36,1,1,1,1,0,0)

tnet["fc6"] = nn.Linear(18432,4096) #512*6*6
transferW(tnet["fc6"].weight, net.params['fc6'][0].data)
transferB(tnet["fc6"].bias, net.params['fc6'][1].data)
tnet["relu6"] = nn.ReLU()
tnet["drop6"] = nn.Dropout(0.5)

tnet["fc7"] = nn.Linear(4096,1024)
transferW(tnet["fc7"].weight, net.params['fc7'][0].data)
transferB(tnet["fc7"].bias, net.params['fc7'][1].data)
tnet["relu7"] = nn.ReLU()
tnet["drop7"] = nn.Dropout(0.5)

tnet["cls_score"] = nn.Linear(1024,21)

tnet["bbox_pred"] = nn.Linear(1024,84)

print net.params
PyTorchAug.save("vgg1024.t7",tnet)
