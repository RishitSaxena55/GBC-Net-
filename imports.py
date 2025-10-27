import torch
from torch import nn
import torchvision
from torchvision import ops
from torch import optim
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.ops import roi_pool