import numpy as np
from setuptools import setup, find_packages, Extension
from Cython.Build import cythonize

extensions = [
    Extension("binsim.utils.graph.myGraph",
              sources=["lib/binsim/utils/graph/myGraph.pyx"]),
    Extension("binsim.disassembly.ida.cfg.util.pyIDAIns",
              sources=["lib/binsim/disassembly/ida/cfg/util/pyIDAIns.pyx",
                       "lib/binsim/disassembly/ida/cfg/util/IDAIns.cpp"],
              language="c++"),
    Extension("binsim.neural.utils.data.dataset.InsCFG.basic_block_chunk_proxy",
              sources=["lib/binsim/neural/utils/data/dataset/InsCFG/basic_block_chunk_proxy.pyx"]),
    Extension("binsim.disassembly.binaryninja.core.fast_utils",
              sources=["lib/binsim/disassembly/binaryninja/core/fast_utils.pyx"],
              include_dirs=[np.get_include()]),
    Extension("binsim.neural.lm.asm2vec.fast_utils",
              sources=["lib/binsim/neural/lm/asm2vec/fast_utils.pyx"])
]

package_data = {
    "binsim.disassembly.binaryninja.core.graph": ["data/mnem/*/*"],
    "binsim.disassembly.ida.ast": ["extra_scripts/*"],
    "binsim.disassembly.ida.cfg": ["extra_scripts/*"],
    "binsim.disassembly.ida.cfg.util": ["data/tokenizer/*"]
}

setup(
    name="binsim",
    version="0.0.0",
    description="A simple python package for binary code similarity detection.",
    author="anonymous",
    packages=find_packages("lib"),
    package_dir={"": "lib"},
    keywords="Binary Analysis, Deep Learning",
    ext_modules=cythonize(extensions),
    package_data= package_data
)

if __name__ == '__main__':
    pass
