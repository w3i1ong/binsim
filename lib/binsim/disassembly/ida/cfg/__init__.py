import os
from ..script import Script
from .JTransSeq import JTransSeq
script = Script(script_path=os.path.split(os.path.abspath(__file__))[0]+'/extra_scripts')
