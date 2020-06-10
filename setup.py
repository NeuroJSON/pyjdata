from setuptools import setup

with open("README.md", "r") as fh:
    readme = fh.read()

setup(
  name = 'jdata',
  packages = ['jdata'],
  version = '0.3.0',
  license='Apache license 2.0',
  description = 'Encoding and decoding Python data structrues using portable JData-annotated formats',
  long_description=readme,
  long_description_content_type="text/markdown",
  author = 'Qianqian Fang',
  author_email = 'fangqq@gmail.com',
  maintainer= 'Qianqian Fang',
  url = 'https://github.com/fangq/pyjdata',
  download_url = 'https://github.com/fangq/pyjdata/archive/v0.3.tar.gz',
  keywords = ['JSON', 'JData', 'UBJSON', 'OpenJData', 'NeuroJData', 'JNIfTI', 'Encoder', 'Decoder'],
  platforms="any",
  install_requires=[
      ],
  classifiers=[
    'Development Status :: 3 - Alpha',
    'Intended Audience :: Developers',
    'Topic :: Software Development :: Build Tools',
    'License :: OSI Approved :: Apache Software License',
    'Programming Language :: Python :: 2.7',
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.4',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6'
  ]
)
