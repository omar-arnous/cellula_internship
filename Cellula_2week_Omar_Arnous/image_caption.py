import os
from PIL import Image
import matplotlib.pyplot as plt
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch
import warnings
import logging
from transformers.utils import logging as hf_logging