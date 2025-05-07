""" 
This is for MEDoE mac, add temp doe coef to action selection
"""

from modules.agents import REGISTRY as agent_REGISTRY
from components.action_selectors import REGISTRY as action_REGISTRY
import torch as th

from .basic_controller import BasicMAC

# This multi-agent controller shares parameters between agents

class DoEMAC(BasicMAC):
    def __init__(self, scheme, groups, args):
        super(DoEMAC, self).__init__(scheme, groups, args)
        # add doe classifier
        self.ent_coef = 1.0 # needed to override the ent_coef called elsewhere

        # self.base_temp = self.args.get("base_temp", 1.0)
        self.base_temp = getattr(self.args, "base_temp", 1.0)
        # self.base_temp = self.args.base_temp if hasattr(self.args, "base_temp") else 1.0
        # self.boost_temp_coef = self.args.get("boost_temp", 1.0)
        self.boost_temp_coef = getattr(self.args, "boost_temp", 1.0)
        

    def is_doe(self, obs, agent_id=None):
        return self.doe_classifier.is_doe(obs, agent_id=agent_id)


    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False):
        # Only select actions for the selected batch elements in bs
        avail_actions = ep_batch["avail_actions"][:, t_ep]

        agent_outputs = self.forward(ep_batch, t_ep, test_mode=test_mode)

        # 修改 forward 为 DoE gain
        obs = ep_batch["obs"][:, t_ep]
        if not test_mode:
            agent_outputs = agent_outputs/self.boost_temp(obs).to(agent_outputs.device)
        else:
            agent_outputs = agent_outputs

        chosen_actions = self.action_selector.select_action(agent_outputs[bs], avail_actions[bs], t_env, test_mode=test_mode)
        return chosen_actions
    

    def forward(self, ep_batch, t, test_mode=False):
        agent_inputs = self._build_inputs(ep_batch, t)
        avail_actions = ep_batch["avail_actions"][:, t]
        agent_outs, self.hidden_states = self.agent(agent_inputs, self.hidden_states)

        # Softmax the agent outputs if they're policy logits
        if self.agent_output_type == "pi_logits":

            if getattr(self.args, "mask_before_softmax", True):
                # Make the logits for unavailable actions very negative to minimise their affect on the softmax
                reshaped_avail_actions = avail_actions.reshape(ep_batch.batch_size * self.n_agents, -1)
                agent_outs[reshaped_avail_actions == 0] = -1e10
            agent_outs = th.nn.functional.softmax(agent_outs, dim=-1)

        return agent_outs.view(ep_batch.batch_size, self.n_agents, -1)
    

    def init_hidden(self, batch_size):
        self.hidden_states = self.agent.init_hidden().unsqueeze(0).expand(batch_size, self.n_agents, -1)  # bav

    def parameters(self):
        return self.agent.parameters()

    def load_state(self, other_mac):
        self.agent.load_state_dict(other_mac.agent.state_dict())

    def cuda(self):
        self.agent.cuda()

    def save_models(self, path):
        th.save(self.agent.state_dict(), "{}/agent.th".format(path))

    def load_models(self, path):
        """
        改为load actor_init
        """
        # self.agent.load_state_dict(th.load("{}/agent.th".format(path), map_location=lambda storage, loc: storage))
        self.agent.load_state_dict(th.load("{}/actor_init.th".format(path), map_location=lambda storage, loc: storage))

    def _build_agents(self, input_shape):
        self.agent = agent_REGISTRY[self.args.agent](input_shape, self.args)

    # def _build_inputs(self, batch, t):
    #     # Assumes homogenous agents with flat observations.
    #     # Other MACs might want to e.g. delegate building inputs to each agent
    #     bs = batch.batch_size
    #     inputs = []
    #     inputs.append(batch["obs"][:, t])  # b1av
    #     if self.args.obs_last_action:
    #         if t == 0:
    #             inputs.append(th.zeros_like(batch["actions_onehot"][:, t]))
    #         else:
    #             inputs.append(batch["actions_onehot"][:, t-1])
    #     if self.args.obs_agent_id:
    #         inputs.append(th.eye(self.n_agents, device=batch.device).unsqueeze(0).expand(bs, -1, -1))
    #
    #     inputs = th.cat([x.reshape(bs*self.n_agents, -1) for x in inputs], dim=1)
    #     return inputs
    #
    # def _get_input_shape(self, scheme):
    #     input_shape = scheme["obs"]["vshape"]
    #     if self.args.obs_last_action:
    #         input_shape += scheme["actions_onehot"]["vshape"][0]
    #     if self.args.obs_agent_id:
    #         input_shape += self.n_agents
    #
    #     return input_shape

    # 改成固定长度 one-hot
    def _build_inputs(self, batch, t):
        # Assumes homogenous agents with flat observations.
        # Other MACs might want to e.g. delegate building inputs to each agent
        bs = batch.batch_size
        inputs = []
        inputs.append(batch["obs"][:, t])  # b1av
        if self.args.obs_last_action:
            if t == 0:
                inputs.append(th.zeros_like(batch["actions_onehot"][:, t]))
            else:
                inputs.append(batch["actions_onehot"][:, t-1])
        if self.args.obs_agent_id:
            fixed_len = 5  # 固定为长度为 5 的 one-hot
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
            # 固定为长度为 5 的 one-hot
            input_shape += 5

        return input_shape
    
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
        self.doe_classifier = classifier
