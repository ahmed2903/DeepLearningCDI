# ###########################################
# Filename: cnnphase.py
# Convenience re-export — imports everything from the split modules
# so that existing scripts (runTrain.py, runYMO.py) work unchanged.
#
# Authors: Marcus Newton, Ahmed H. Mokhtar.
# Licence: GNU GPL 3
# ###########################################

from model import NNModel
from train import CNNTrain
from predict import ShrinkWrap, CNNPredict
