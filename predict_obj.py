from cnn import *

predict = CNNPredict()
predict.SetDeviceType('cuda')
predict.SetModel(NNModel)
predict.SetExpData('expdata.py', mask=500)
predict.SetTrainedNN('CP200.pth')
predict.SetOutputFile('output.py')
predict.Predict()
