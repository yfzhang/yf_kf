#from distutils.core import setup
#
#setup(name='Distutils',
#      version='1.0',
#      description='Python Distribution Utilities',
#      author='Greg Ward',
#      author_email='gward@python.net',
#      url='https://www.python.org/sigs/distutils-sig/',
#      packages=['distutils', 'distutils.command'],
#     )

from setuptools import setup

setup(name='yf_kf',
      version='0.0',
      url='https://github.com/yfzhang/yf_kf',
      license='BSD',
      author='yfzhang',
      author_email='zhan0314@gmail.com',
      description='customized kalman filtering implementation',
      packages=['yf_kf'],
      zip_safe=False)

