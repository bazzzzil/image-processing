# image-processing #

This repo is a collection of image processing techniques I've learned over the years, mostly from a Fourier analysis lens.

`impro_utils.py` is a module script file containing functions that I've found useful in image processing. Much of these can be found in the OpenCV library, but it's always good to know how it all works under the hood :)

`example.ipynb` and `fourier_demo.ipynb` are Jupyter notebook files that apply the functions from `impro_utils.py` to showcase some imaging processing techniques. Example image files, `lena.mat` and `test2.jpg`, are provided.

`svd_compression.ipynb` is a notebook that demonstrates SVD-based image compression. `dct_compression.ipynb` is a notebook that demonstrates DCT-based image compression. `image_segmentation.ipynb` showcases traditional image segmentation techniques.

`img_segment_nn.ipynb`is a notebook that demonstrates the use of Transformers in image segmentation and compares with the traditional techniques. 

WIP: 
`wavelet_compression.ipynb` (as of Sep 11 2026)
Basis function visualization for DCT compression (complete as of Sep 14 2026)
Deeper discussion on FT vs FFT for Fourier transform (as of Sep 11 2026)