import sys

sys.dont_write_bytecode = True

# import pytest

from scripts import train

from ray.rllib.algorithms import AlgorithmConfig

# @pytest.mark.skip
# @pytest.mark.parametrize("arg_name, invalid_input", [
#         ("--num-gpus",[-1,"1",None])
#     ]
# )
# def test_script_train_parser():
#     pass


def test_script_config():
    out = train.get_algorithm_config()
    assert isinstance(out, AlgorithmConfig)


# Add torch, ray, "ray[tune]", Pillow

# Fix ENV registry
