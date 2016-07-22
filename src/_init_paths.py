"""
Set up paths for Video Processing Pipeline.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
# from __future__ import unicode_literals
import os
import sys


def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)


this_dir = os.path.dirname(__file__)

# Add lib to PATH
lib_path = os.path.join(this_dir, '..', 'lib')
add_path(lib_path)
lib_path = os.path.join(this_dir, '..', 'lib/pyflow')
add_path(lib_path)
lib_path = os.path.join(this_dir, '..', 'lib/pydensecrf')
add_path(lib_path)
