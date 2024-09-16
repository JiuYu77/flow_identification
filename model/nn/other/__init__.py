# -*- coding: UTF-8 -*-
import sys
from pathlib import Path
FILE = Path(__file__).resolve()
root = FILE.parents[3]
nn = FILE.parents[0]
sys.path.append(str(nn))

from model.nn.other.block2 import *
