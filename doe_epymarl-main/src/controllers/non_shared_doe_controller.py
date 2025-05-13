""" 
This is for Non-shared MEDoE mac, combining non-shared architecture with DoE functionality
"""

from modules.agents import REGISTRY as agent_REGISTRY
from components.action_selectors import REGISTRY as action_REGISTRY
import torch as th

from .non_shared_controller import NonSharedMAC

class DoENonSharedMAC(NonSharedMAC):
    def __init__(self, scheme, groups, args):
        super(DoENonSharedMAC, self).__init__(scheme, groups, args)
        # add doe classifier
        self.ent_coef = 1.0 
        
        self.base_temp = getattr(self.args, "base_temp", 1.0)
        self.boost_temp_coef = getattr(self.args, "boost_temp", 1.0)

    def is_doe(self, obs, agent_id=None):
        return self.doe_classifier.is_doe(obs, agent_id=agent_id)

    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False):
        # Only select actions for the selected batch elements in bs
        avail_actions = ep_batch["avail_actions"][:, t_ep]
        agent_outputs = self.forward(ep_batch, t_ep, test_mode=test_mode)

        # 修改 forward 为 DoE tmp gain
        obs = ep_batch["obs"][:, t_ep]
        if not test_mode:
            agent_outputs = agent_outputs/self.boost_temp(obs).to(agent_outputs.device)
        else:
            agent_outputs = agent_outputs

        chosen_actions = self.action_selector.select_action(agent_outputs[bs], avail_actions[bs], t_env, test_mode=test_mode)
        return chosen_actions

    # def load_models(self, path):
    #     """
    #     改为load actor_init
    #     从一个大的模型中load 每个agent的actor_init？
    #     """
    #     self.agent.load_state_dict(th.load("{}/actor_init.th".format(path), map_location=lambda storage, loc: storage))

    # def load_models(self, path):
    #     actor_models = th.load(f"{path}/actor_init.th", map_location=lambda storage, loc: storage)
    #
    #     # 遍历每个 agent，并加载各自参数
    #     for agent_id in range(self.n_agents):
    #         agent_state_dict = actor_models[agent_id]
    #
    #         # 过滤掉前缀（如 agents.0. → 去掉前缀）
    #         new_state_dict = {}
    #         for k, v in agent_state_dict.items():
    #             # 移除 agents.{id}. 前缀
    #             new_key = k.replace(f"agents.{agent_id}.", "")
    #             new_state_dict[new_key] = v
    #
    #
    #         self.agent[agent_id].load_state_dict(new_state_dict)
    #
    #     print(f"Loaded actor models for {self.n_agents} agents.")
    #
    # def load_models(self, path):
    #     actor_models = th.load(f"{path}/actor_init.th", map_location=lambda storage, loc: storage)
    #
    #     for agent_id in range(self.n_agents):
    #         agent_state_dict = actor_models[agent_id]
    #
    #         new_state_dict = {
    #             k.replace(f"agents.{agent_id}.", ""): v
    #             for k, v in agent_state_dict.items()
    #         }
    #
    #         self.agent.agents[agent_id].load_state_dict(new_state_dict)
    #
    #     print(f"Loaded actor models for {self.n_agents} agents.")

    def load_models(self, path):
        # 加载包含多个 OrderedDict 的列表
        actor_models = th.load(f"{path}/actor_init.th", map_location=lambda storage, loc: storage)

        # 确保加载的模型和 agent 数量匹配
        assert len(
            actor_models) == self.n_agents, f"Loaded models count does not match the number of agents: {len(actor_models)} != {self.n_agents}"

        # 遍历每个 agent
        for agent_id in range(self.n_agents):
            # 当前 agent 的状态字典
            agent_state_dict = actor_models[agent_id]
            print("keys", agent_state_dict.keys())

            # 遍历 agent 的 state_dict，移除前缀并将其加载到当前 agent 中
            new_state_dict = {
                k.replace(f"agents.{agent_id}.", ""): v
                for k, v in agent_state_dict.items()
            }

            # 加载到该 agent 的模型
            self.agent.agents[agent_id].load_state_dict(new_state_dict)

        print(f"Loaded actor models for {self.n_agents} agents.")

    # def load_models(self, path):
    #     actor_models = th.load(f"{path}/actor_init.th", map_location=lambda storage, loc: storage)
    #
    #     for agent_id in range(self.n_agents):
    #         agent_state_dict = actor_models[agent_id]
    #
    #         # 去掉 "agents.{id}." 的前缀
    #         new_state_dict = {
    #             k.replace(f"agents.{agent_id}.", ""): v
    #             for k, v in agent_state_dict.items()
    #         }
    #
    #         model = self.agent.agents[agent_id]
    #         model_state_dict = model.state_dict()
    #
    #         # 过滤掉 shape 不匹配的参数
    #         filtered_state_dict = {}
    #         skipped_keys = []
    #         for k, v in new_state_dict.items():
    #             if k in model_state_dict:
    #                 if v.shape == model_state_dict[k].shape:
    #                     filtered_state_dict[k] = v
    #                 else:
    #                     skipped_keys.append(k)
    #             else:
    #                 skipped_keys.append(k)
    #
    #         # 加载过滤后的参数
    #         model.load_state_dict(filtered_state_dict, strict=False)
    #
    #         if skipped_keys:
    #             print(f"[Agent {agent_id}] Skipped {len(skipped_keys)} unmatched keys:")
    #             for key in skipped_keys:
    #                 print(f"  - {key} (shape: {new_state_dict[key].shape})")
    #
    #     print(f"Loaded actor models for {self.n_agents} agents (with shape checks).")

    """ Used for adjust policy temperature """
    def boost_temp(self, obs, agent_id=None):
        if agent_id is None:
            labels = th.stack([self.boost_temp(obs, agent_id) for agent_id in range(self.n_agents)])
            labels_align = labels.permute(1,0,2)
            return labels_align
        else:
            doe = self.is_doe(obs[:, agent_id, :], agent_id)
            return self.base_temp * th.pow(self.boost_temp_coef, 1-doe)
    
    # Add DoE Classifier
    def set_doe_classifier(self, classifier):
        assert len(classifier.mlps) == self.n_agents
        self.doe_classifier = classifier

    # 改成固定长度 one-hot
    def _build_inputs(self, batch, t):
        bs = batch.batch_size
        inputs = []
        inputs.append(batch["obs"][:, t])  # b1av
        if self.args.obs_last_action:
            if t == 0:
                inputs.append(th.zeros_like(batch["actions_onehot"][:, t]))
            else:
                inputs.append(batch["actions_onehot"][:, t-1])

        # 固定长度 one-hot，考虑从args中直接读取对应的agent id和onehot编码？
        if self.args.obs_agent_id:
            fixed_len  = getattr(self.args, "total_agents", 5)  # 默认值5，可在config中设置
            eye_matrix = th.zeros(self.n_agents, fixed_len, device=batch.device)
            eye_matrix[:, :self.n_agents] = th.eye(self.n_agents, device=batch.device)
            inputs.append(eye_matrix.unsqueeze(0).expand(bs, -1, -1))  # shape: (bs, n_agents, fixed_len)


        inputs = th.cat([x.reshape(bs*self.n_agents, -1) for x in inputs], dim=1)
        return inputs

    def _get_input_shape(self, scheme):
        input_shape = scheme["obs"]["vshape"]
        if self.args.obs_last_action:
            input_shape += scheme["actions_onehot"]["vshape"][0]
        if self.args.obs_agent_id:
            fixed_len  = getattr(self.args, "total_agents", 5)  # 默认值5，可在config中设置
            input_shape += fixed_len

        return input_shape