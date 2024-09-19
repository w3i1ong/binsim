from collections import namedtuple

class Script:
    def __init__(self, script_path, entry_point='main.py'):
        self._script_path = script_path
        self._entry_point = entry_point

    @property
    def entry_point(self):
        return f'{self._script_path}/{self._entry_point}'

