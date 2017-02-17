from setuptools import setup, find_packages


def readme():
    with open('README.rst') as f:
        return f.read()

setup(name='deep_qa',
      version='0.1',
      description='Using deep learning to answer Aristo\'s science questions',
      long_description=readme(),
      classifiers=[
          'Development Status :: 3 - Alpha',
          'License :: OSI Approved :: Apache Software License',
          'Programming Language :: Python :: 3.5',
          'Topic :: Scientific/Engineering :: Artificial Intelligence',
      ],
      keywords='deep_qa NLP deep learning machine reading',
      url='https://github.com/allenai/deep_qa',
      author='Matt Gardner',
      author_email='mattg@allenai.org',
      license='Apache',
      packages=find_packages(),
      install_requires=[
          'keras>=1.2.2',
          'h5py',
          'scikit-learn',
          'theano',
      ],
      setup_requires=['pytest-runner'],
      tests_require=['pytest'],
      include_package_data=True,
      zip_safe=False)
