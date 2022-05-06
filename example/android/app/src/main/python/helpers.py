import sys
import datetime
from os.path import join, dirname
from os import listdir

def get_version_info():
  version = sys.version_info
  now = datetime.datetime.utcnow()
  print("Running Python {0}.{1}.{2}".format(version.major, version.minor, version.micro))

def read_file(name):
  print("\n[python] Read file...")
  print(listdir(dirname(__file__)))
  filename = join(dirname(__file__), f'{name}.py')
  print(filename) 
  try:
    with open(filename, 'r') as f:
      contents = f.read()
      print(contents)
      return contents
  except Exception as e:
    print(e)
    return ""