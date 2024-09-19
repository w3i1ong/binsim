import numpy as np
import torch
from glob import glob
from Cython.Build import cythonize
from pybind11.setup_helpers import Pybind11Extension
from setuptools import setup, find_packages, Extension
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

extensions = [
    Extension("binsim.utils.graph.myGraph",
              sources=["lib/binsim/utils/graph/myGraph.pyx"]),
    Extension("binsim.disassembly.backend.binaryninja.fast_utils",
              sources=["lib/binsim/disassembly/backend/binaryninja/fast_utils.pyx"],
              include_dirs=[np.get_include()]),
    Extension("binsim.neural.lm.asm2vec.fast_utils",
              sources=["lib/binsim/neural/lm/asm2vec/fast_utils.pyx"])
]

pybind11_extensions = [
    Pybind11Extension("binsim.disassembly.backend.binaryninja.graph.datautils",
                      sources=["lib/binsim/disassembly/backend/binaryninja/graph/src/main.cpp"]+
                             glob("lib/binsim/disassembly/backend/binaryninja/graph/src/graph/*.cpp")+
                             glob("lib/binsim/disassembly/backend/binaryninja/graph/src/utils/*.cpp")),

    Pybind11Extension("binsim.disassembly.backend.ida.graph.jTransSeq.util.IDAIns",
              sources=["lib/binsim/disassembly/backend/ida/graph/jTransSeq/util/IDAIns.cpp"]),
]

if torch.cuda.is_available():
    torch_extensions = [
        CUDAExtension('binsim.neural.nn.layer.dagnn.dagrnn_ops', [
            "lib/binsim/neural/nn/layer/dagnn/dagrnn_ops/daggru.cu",
            "lib/binsim/neural/nn/layer/dagnn/dagrnn_ops/tree_lstm.cu",
            "lib/binsim/neural/nn/layer/dagnn/dagrnn_ops/graph.cpp",
            "lib/binsim/neural/nn/layer/dagnn/dagrnn_ops/main.cpp"
        ], extra_compile_args={'cxx': ['-O3'], 'nvcc': ['-O3']})
    ]
else:
    torch_extensions = []
    print("CUDA is not available, skip building CUDA extensions.")

package_data = {
    "binsim.disassembly.backend.binaryninja.graph": ["data/mnem/*/*"],
    "binsim.disassembly.backend.ida.graph.jTransSeq.util": ["data/tokenizer/*"],
    "binsim.disassembly.backend.ida" : ["extra_scripts/*"]
}

setup(
    name="binsim",
    version="0.2.0",
    description="A simple python package for binary code similarity detection.",
    author="w3i1ong",
    author_email="liwl23@mail2.sysu.edu.cn",
    packages=find_packages("lib"),
    package_dir={"": "lib"},
    keywords="Binary Analysis, Deep Learning",
    ext_modules=cythonize(extensions) + pybind11_extensions + torch_extensions,
    package_data= package_data,
    install_requires=[
        'torch',
    ],
    cmdclass={
        'build_ext': BuildExtension
    },
    python_requires='>=3.10',
)

if __name__ == '__main__':
    pass
