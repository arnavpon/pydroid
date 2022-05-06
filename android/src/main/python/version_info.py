import sys
import datetime

def get_version_info():
  version = sys.version_info
  now = datetime.datetime.utcnow()
  print("Running Python {0}.{1}.{2}".format(version.major, version.minor, version.micro))

get_version_info()