import sys
import pyximport

pyximport.install()
if sys.platform == 'linux':
    from .fast_utils import control_reachable_distance_matrix, compute_function_hash
elif sys.platform == 'win32':
    def control_reachable_distance_matrix(*args, **kwargs):
        raise NotImplementedError("control_reachable_distance_matrix is Not implemented for Windows")

    def compute_function_hash(*args, **kwargs):
        raise NotImplementedError("compute_function_hash is Not implemented for Windows")
