import os.path

from .codeast import Tree, CodeAST
from ..script import Script

script = Script(script_path=os.path.split(os.path.abspath(__file__))[0]+'/extra_scripts')
