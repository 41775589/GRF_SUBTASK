from .joint_mlp_class import JointMLPClassifier
from .mlp_class import MLPClassifier
from .llm_class import LLMlassifier

REGISTRY = {}

REGISTRY["mlp"] = MLPClassifier
REGISTRY["joint_mlp"] = JointMLPClassifier
REGISTRY["llm_mlp"] = LLMlassifier


def doe_classifier_config_loader(n_agents, cfg, buffer_path, load_mode):
    type = cfg.get('doe_type', 'mlp')
    cls = REGISTRY[type]
    if not hasattr(cls, "from_config"):
        raise NotImplementedError(f"There is no from_config method defined for {type}")
    else:
        return cls.from_config(n_agents, cfg, buffer_path, load_mode)
