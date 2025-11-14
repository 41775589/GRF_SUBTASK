"""
Reward Wrapper Registry for CookingZoo
Allows dynamic loading of reward functions from config files.

Directory structure:
    reward_wrappers/
    ├── __init__.py
    ├── reward_wrapper_registry.py  (this file)
    ├── base_wrapper.py
    ├── default_reward.py
    ├── cooperation_reward.py
    ├── shaping_reward.py
    └── your_custom_reward.py
"""

import importlib
import os
from typing import Dict, Type, Optional
from pettingzoo.utils.wrappers import BaseWrapper

# Global registry for reward wrappers
_REWARD_WRAPPER_REGISTRY: Dict[str, Type[BaseWrapper]] = {}


def register_reward_wrapper(name: str):
    """
    Decorator to register a reward wrapper class.

    Usage:
        @register_reward_wrapper("my_custom_reward")
        class MyCustomRewardWrapper(BaseWrapper):
            ...
    """

    def decorator(cls):
        if name in _REWARD_WRAPPER_REGISTRY:
            print(f"Warning: Overwriting reward wrapper '{name}'")
        _REWARD_WRAPPER_REGISTRY[name] = cls
        return cls

    return decorator


def get_reward_wrapper(name: str) -> Type[BaseWrapper]:
    """
    Get a reward wrapper class by name.

    Args:
        name: Name of the registered reward wrapper

    Returns:
        The reward wrapper class

    Raises:
        KeyError: If wrapper name not found
    """
    if name not in _REWARD_WRAPPER_REGISTRY:
        raise KeyError(
            f"Reward wrapper '{name}' not found. "
            f"Available wrappers: {list(_REWARD_WRAPPER_REGISTRY.keys())}"
        )
    return _REWARD_WRAPPER_REGISTRY[name]


def list_reward_wrappers():
    """List all registered reward wrappers."""
    return list(_REWARD_WRAPPER_REGISTRY.keys())


def load_reward_wrapper_from_file(filepath: str, wrapper_name: Optional[str] = None):
    """
    Dynamically load a reward wrapper from a Python file.

    Args:
        filepath: Path to the Python file (e.g., 'reward_wrappers/my_reward.py')
        wrapper_name: Optional name to register the wrapper under.
                     If None, uses the filename without extension.

    Returns:
        The loaded wrapper class

    Example:
        load_reward_wrapper_from_file('custom_rewards/my_reward.py', 'my_reward')
        wrapper_cls = get_reward_wrapper('my_reward')
    """
    # Get module name from filepath
    module_name = os.path.splitext(os.path.basename(filepath))[0]

    # Load the module
    spec = importlib.util.spec_from_file_location(module_name, filepath)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {filepath}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # Find wrapper classes in the module
    wrapper_classes = []
    for attr_name in dir(module):
        attr = getattr(module, attr_name)
        if (isinstance(attr, type) and
                issubclass(attr, BaseWrapper) and
                attr is not BaseWrapper):
            wrapper_classes.append((attr_name, attr))

    if not wrapper_classes:
        raise ValueError(f"No reward wrapper class found in {filepath}")

    # Use the first wrapper class found
    cls_name, wrapper_cls = wrapper_classes[0]

    # Register it
    register_name = wrapper_name or module_name
    _REWARD_WRAPPER_REGISTRY[register_name] = wrapper_cls

    print(f"Loaded reward wrapper '{register_name}' from {filepath}")
    return wrapper_cls


def create_reward_wrapper(env, wrapper_name: str, wrapper_config: Optional[Dict] = None):
    """
    Create a reward wrapper instance.

    Args:
        env: The base environment to wrap
        wrapper_name: Name of the registered wrapper or path to Python file
        wrapper_config: Configuration dict to pass to the wrapper

    Returns:
        Wrapped environment

    Example:
        # Using registered wrapper
        wrapped_env = create_reward_wrapper(env, "cooperation_reward",
                                           {"cooperation_bonus": 5.0})

        # Using file path
        wrapped_env = create_reward_wrapper(env, "path/to/my_reward.py",
                                           {"param": 1.0})
    """
    wrapper_config = wrapper_config or {}

    # Check if wrapper_name is a file path
    if wrapper_name.endswith('.py') and os.path.exists(wrapper_name):
        load_reward_wrapper_from_file(wrapper_name)
        wrapper_name = os.path.splitext(os.path.basename(wrapper_name))[0]

    # Get the wrapper class
    wrapper_cls = get_reward_wrapper(wrapper_name)

    # Create and return wrapped environment
    return wrapper_cls(env, **wrapper_config)


# Auto-register wrappers from reward_wrappers directory
def auto_register_wrappers(directory: str = "reward_wrappers"):
    """
    Automatically register all wrapper classes from Python files in a directory.

    Args:
        directory: Directory containing reward wrapper Python files
    """
    if not os.path.exists(directory):
        return

    for filename in os.listdir(directory):
        if filename.endswith('.py') and not filename.startswith('_'):
            filepath = os.path.join(directory, filename)
            try:
                load_reward_wrapper_from_file(filepath)
            except Exception as e:
                print(f"Warning: Failed to load {filepath}: {e}")


# Initialize: auto-register wrappers if directory exists
try:
    auto_register_wrappers()
except Exception as e:
    print(f"Warning: Auto-registration failed: {e}")