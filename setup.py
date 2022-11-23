from setuptools import setup

with open("README.md", "r") as fh:
    readme = fh.read()

setup(
  name = 'jdata',
  packages = ['jdata'],
  version = '0.5.2',
  license='Apache license 2.0',
  description = 'Encoding and decoding Python data structrues using portable JData-annotated formats',
  long_description=readme,
  long_description_content_type="text/markdown",
  author = 'Qianqian Fang',
  author_email = 'fangqq@gmail.com',
  maintainer= 'Qianqian Fang',
  url = 'https://github.com/NeuroJSON/pyjdata',
  download_url = 'https://github.com/NeuroJSON/pyjdata/archive/v0.5.2.tar.gz',
  keywords = ['JSON', 'JData', 'UBJSON', 'BJData', 'OpenJData', 'NeuroJSON', 'JNIfTI', 'JMesh', 'Encoder', 'Decoder'],
  platforms="any",
  install_requires=[
        'numpy>=1.8.0'
      ],
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Software Development :: Build Tools',
    'License :: OSI Approved :: Apache Software License',
    'Programming Language :: Python :: 2.7',
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.4',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3.7',
    'Programming Language :: Python :: 3.8',
    'Programming Language :: Python :: 3.9',
    'Programming Language :: Python :: 3.10',
    'Topic :: Software Development :: Libraries',
    'Topic :: Software Development :: Libraries :: Python Modules'
  ]
)
