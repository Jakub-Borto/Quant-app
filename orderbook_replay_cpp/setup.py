from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

setup(
    ext_modules=[
        Pybind11Extension(
            "orderbook_replay_cpp",
            ["src/orderbook_replay.cpp"],
            cxx_std=20,
        )
    ],
    cmdclass={"build_ext": build_ext},
)
