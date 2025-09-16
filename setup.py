from setuptools import setup, find_packages


setup(name='Purification',
      version='1.0.0',
      description='Semantic Information fused into Feature and Label Purification',
      author=['Nethra', 'Pradyumna'],
      author_email='',
      # url='',
      install_requires=[
          'numpy', 'torch', 'torchvision',
          'six', 'h5py', 'Pillow', 'scipy',
          'scikit-learn', 'metric-learn', 'faiss_gpu'],
      packages=find_packages(),
      keywords=[
          'Unsupervised Learning',
          'Contrastive Learning',
          'Person Re-identification'
      ])
