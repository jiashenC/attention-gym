from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

setup(
    name="custom_attn_kernel",
    ext_modules=[
        CUDAExtension(
            name="custom_attn_kernel",
            sources=["custom_attn_kernel.cu"],
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
