from setuptools import setup
from torch.utils import cpp_extension

setup(
    name="quantize_int4",
    ext_modules=[cpp_extension.CUDAExtension("quantize_int4", ["quantize_int4.cu"])],
    cmdclass={"build_ext": cpp_extension.BuildExtension},
)
