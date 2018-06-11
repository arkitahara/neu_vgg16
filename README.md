# neu_vgg16
README

Thanks for your interest in this tool! To help you get started, you'll need (and may want) Python 3.5+ and the following libraries:
- numpy
- matplotlib
- scikit-image (skimage)
- scikit-learn (sklearn)
- keras
- tensorflow
- jupyter
- ntsne wrapper (recommended: Brian DeCost, https://github.com/bdecost/ntsne)

A sample of this code executed on the test data set is shown in neu_vgg16.html

This code was prepared using Jupyter and is known to run in that environment.

Please find in this package:
- A jupyter notebook including code for this project
- HTML output showing a demonstration of proper code execution

USAGE:
To run this code, you'll need a directory of image files (default type .png). Set that directory in the first code block.
Note: it's best to normalize your images before running this algorithm. You can use your favorite normalization scheme; we used adaptive equalization in the scikit-image library.
Make a directory to store all results and set as the variable "results". However, this will execute properly with non-normalized images in the NEU Surface Defect Database too.

Dataset:
The NEU Surface Defect Database was developed by K. Song and Y. Yan. Please refer to their website for database information.
http://faculty.neu.edu.cn/yunhyan/NEU_surface_defect_database.html
K. Song and Y. Yan, “A noise robust method based on completed local binary patterns for hot-rolled steel strip surface defects,” Applied Surface Science, vol. 285, pp. 858-864, Nov. 2013.(paper)
