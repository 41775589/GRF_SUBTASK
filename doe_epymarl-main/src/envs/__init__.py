import os
import sys

from smac.env import MultiAgentEnv, StarCraft2Env

from .gymma import GymmaWrapper

# 在envs/__init__.py中添加
from .overcooked import overcookedEnv


def smac_fn(**kwargs) -> MultiAgentEnv:
    assert "common_reward" in kwargs and "reward_scalarisation" in kwargs
    assert kwargs[
        "common_reward"
    ], "SMAC only supports common reward. Please set `common_reward=True` or choose a different environment that supports general sum rewards."
    del kwargs["common_reward"]
    del kwargs["reward_scalarisation"]
    return StarCraft2Env(**kwargs)


def gymma_fn(**kwargs) -> MultiAgentEnv:
    assert "common_reward" in kwargs and "reward_scalarisation" in kwargs
    return GymmaWrapper(**kwargs)


REGISTRY = {}
REGISTRY["sc2"] = smac_fn

if sys.platform == "linux":
    os.environ.setdefault(
        "SC2PATH", os.path.join(os.getcwd(), "3rdparty", "StarCraftII")
    )

REGISTRY["gymma"] = gymma_fn


# Add GRF API
from functools import partial
# import sys
# import os

# # 获取绝对路径
# project_path = os.path.abspath('/data/qiaodan/projects/GRF_SUBTASK')

# # 将路径添加到 sys.path
# if project_path not in sys.path:
#     sys.path.insert(0, project_path)
    
def env_fn(env, **kwargs) -> MultiAgentEnv:
    # remove common_reward and reward_scalarisation
    kwargs.pop('common_reward', None)
    kwargs.pop('reward_scalarisation', None)
    return env(**kwargs)

try:
    gfootball = True
    from .gfootball import GoogleFootballEnv
    # from gfootball import GoogleFootballEnv  # 调试相对路径失败



except Exception as e:
    gfootball = False
    print(e)


try:
    cooking_zoo = True
    from .cooking_zoo import CookingZooEnv


except Exception as e:
    overcooked = False
    print(e)

try:
    chainball = True
    from .chain_ball import ChainballWrapper
    # from gfootball import GoogleFootballEnv  # 调试相对路径失败
except Exception as e:
    overcooked = False
    print(e)

if gfootball:
    REGISTRY["gfootball"] = partial(env_fn, env=GoogleFootballEnv)

if overcooked:
    REGISTRY["cooking_zoo"] = partial(env_fn, env=CookingZooEnv)

if chainball:
    REGISTRY["chainball"] = partial(env_fn, env=ChainballWrapper)