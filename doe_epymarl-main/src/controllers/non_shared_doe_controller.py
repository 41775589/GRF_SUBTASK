""" 
This is for Non-shared MEDoE mac, combining non-shared architecture with DoE functionality
"""

from modules.agents import REGISTRY as agent_REGISTRY
from components.action_selectors import REGISTRY as action_REGISTRY
import torch as th

from .non_shared_controller import NonSharedMAC

class DoENonSharedMAC(NonSharedMAC):
    def __init__(self, scheme, groups, args):
        # super(NonSharedMAC, self).__init__(scheme, groups, args)
        super().__init__(scheme, groups, args)
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

    def load_models(self, path):
        """
        改为load actor_init
        从一个大的模型中load 每个agent的actor_init？
        """
        self.agent.load_state_dict(th.load("{}/actor_init.th".format(path), map_location=lambda storage, loc: storage))

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