import sys
from os.path import dirname, abspath, join
sys.path.append(abspath(join(dirname(__file__), "../..")))
from .config import *
from .dataloader import *
import sgad.cgn.models as models
