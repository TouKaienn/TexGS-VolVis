from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='compute_normal_cuda',
    ext_modules=[
        CUDAExtension(
            name='compute_normal_cuda',
            sources=['compute_normal_cuda_kernel.cu'],
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
