import pickle
from scipy import io

EXAMPLE_PATH = '/Users/samuelhmorton/indiv_projects/school/masters/pydroid/example/android/app/src/main/python/px1_full (1).mat'


def get_pulseox_array(path):
    """
    Get pulse ox array from .mat file.
    """
    mat = io.loadmat(path)
    return mat['pulseOxRecord'][0]
