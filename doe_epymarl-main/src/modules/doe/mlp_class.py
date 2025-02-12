import os
import os.path as osp
from collections import OrderedDict
import numpy as np
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from .utils import fc_network

from .utils import SimpleListDataset


class MLPClassifier:
    def __init__(self,
                 n_agents,
                 train_dataloader, 
                 test_dataloader,
                 network_arch,
                 role_list,
                 learning_rate=1e-2,
                 batch_size=256,
                 test_period=5,
                 obs_mask=None,
                 ):
        self.n_agents = n_agents
        self.mlps = [fc_network(network_arch) for _ in range(n_agents)]
        self.learning_rates = [learning_rate] * n_agents
        self.network_arch = network_arch
        self.obs_mask = obs_mask
        self.batch_size = batch_size
        self.role_list = role_list

        if train_dataloader is not None:

            # self.trained_agent_id = 0 # 用于单独训练某一个agent doe

            self.train_data_loader = train_dataloader
            self.test_data_loader = test_dataloader
            self.results = self.train_mlp(
                    train_dataloader=train_dataloader,
                    test_dataloader=test_dataloader,
                    role_list=role_list,
                    test_period=test_period,
                    obs_mask=self.obs_mask,
                    # agent_id=self.trained_agent_id
                    )

    def train_mlp(
            self,
            train_dataloader,
            test_dataloader,
            role_list,
            test_period=5,
            obs_mask=None):
        
        results = []

        for agent_id in range(self.n_agents):

            loss_function = torch.nn.BCEWithLogitsLoss()
            optim = Adam(self.mlps[agent_id].parameters(),
                         lr=self.learning_rates[agent_id],
                         eps=1e-8)

            train_results = []
            test_results = []

            # mask某一层，network_arch = [32, 256, 10] 代表网络结构
            if obs_mask is None:
                mask = 1
            else:
                mask = torch.zeros(self.network_arch[0])
                for i in obs_mask:
                    mask[i] = 1

            for batch_idx, (s, label) in enumerate(train_dataloader):
                predicted_label = self.mlps[agent_id](s*mask).flatten()
                egocentric_label = (label == role_list[agent_id]).float()
                # 这里label提供的是0-1代表defend-attack角色的经验，需要经过跟当前agent的role list进行对比，
                # 来判断这个状态是不是符合当前的角色，如果是，那么egocentric_label就是1，否则就是0
                # 所以这里loss_function是BCEWithLogitsLoss，目标是让agent根据状态判断是否符合自身角色
                loss = loss_function(predicted_label, egocentric_label)
                optim.zero_grad()
                loss.backward()
                optim.step()
                train_results.append(loss.item())
                test_loss = 0.0
                if batch_idx % test_period == 0:
                    with torch.no_grad():
                        for s_test, label_test in test_dataloader:
                            predicted_label_test = self.mlps[agent_id](s_test).flatten()
                            egocentric_label_test = (label_test == role_list[agent_id]).float()
                            test_loss += loss_function(predicted_label_test, egocentric_label_test).item()
                        test_results.append(test_loss/len(test_dataloader))
                        # batch是一个递增整数，代表epoch的轮数，多个batch——size的数据

            results.append({ 
                "agent_index": agent_id,
                "train": train_results,
                "test": test_results,
            })
        return results

    # 返回的是obs属于当前角色的0-1概率，需要经过sigmoid函数
    def is_doe(self, obs, agent_id=None):
        device = next(self.mlps[agent_id].parameters()).device  # 获取模型所在的设备
        obs_tensor = torch.Tensor(obs).to(device)  # 将输入张量移动到相同的设备
        if agent_id is None:
            return [self.mlps[i](torch.Tensor(obs_tensor[i])).sigmoid() for i in range(self.n_agents)]
        else:
            return self.mlps[agent_id](torch.Tensor(obs_tensor)).sigmoid()

    # 返回的是mlp输出值，不需要经过sigmoid函数
    def is_doe_logits(self, obs, agent_id=None):
        if agent_id is None:
            return [self.mlps[i](torch.Tensor(obs[i])) for i in range(self.n_agents)]
        else:
            return self.mlps[agent_id](torch.Tensor(obs))

    def update(self):
        ...
    
    def save(self, pathname):
        torch.save(self.mlps, pathname)

    
    @classmethod
    def from_config(cls, n_agents, cfg, buffer_path, load_mode):
        # if cfg.get("load_mode") == "train":
        if load_mode == "train":
            classifier = cls.from_config_train(n_agents, cfg, buffer_path)
            if cfg.get("save_classifier", False):
                save_path = os.path.join(buffer_path, cfg.save_doe_name)
                classifier.save(save_path)
            return classifier
        elif load_mode == "load":
            return cls.from_config_load(n_agents, cfg, buffer_path)
        # elif load_mode == "merge":
        #     return cls.from_merge_config(n_agents, cfg)

    @classmethod
    def from_config_train(cls, n_agents, cfg, buffer_path):
        mlp_cfg = cfg.get("mlp")
        # 初始化role list，用-1代表没有分配角色
        role_list = [-1] * n_agents
        role_ids = cfg.get("role_ids")
        # (0, ('defence', ['alice', 'bob'])) 和 (1, ('attack', ['carol', 'dave']))
        # 这里role_ids是字典，key是角色，value是agent_id

        # 设置角色列表
        for label, (_, role_agents_ids) in enumerate(role_ids.items()):
            for agent_id in role_agents_ids:
                role_list[agent_id] = label
        # role_list = [0, 0, 1, 1, 1] 代表分别是 防御防御进攻进攻进攻，取决于任务num agents和yaml设置

        # buffer_save_path = os.path.join("results", "buffers", mlp_cfg.env, mlp_cfg.env_args.map_name, "buffer.pt")

        """ To LZH
        这里写死的buffer.pt，需要配合多层分解重新命名规则，参考 generator_one_level line 913注释，命名规则一致即可
        """
        if not os.path.exists(buffer_path):
            raise FileNotFoundError(f"Buffer file not found at {buffer_path}")
        exp_buffers = torch.load(os.path.join(buffer_path, "buffer.pt"))

        # 考虑有时候用episode data而不是transitions,getattr
        transition_data = exp_buffers.transition_data
        obs_data = transition_data["obs"]
        buffer_size, max_seq_length, _, obs_shape = obs_data.shape

        # Classifier training params
        batch_size = mlp_cfg.get("batch_size", 256)
        test_fraction = mlp_cfg.get("test_fraction", 0.1)
        hidden_sizes = mlp_cfg.get("hidden_sizes", [128])
        learning_rate = mlp_cfg.get("lr", 1e-2)
        test_period = mlp_cfg.get("test_period", 5)
        obs_mask = mlp_cfg.get("obs_mask", None)

        # Load & process the data
        states = [] 
        labels = [] 

        with torch.no_grad():
            for agent_id in range(n_agents):
                #state = torch.concat(exp_buffers[agent_id])
                # 这里要考虑是否使用全局状态，如果全局状态可见，那么所有agent都一样，没法区分，还是要用obs我觉得，但也要考虑有些环境没有obs
                # state = exp_buffers[agent_id]
                # 这里数据要想办法对齐一下

                # 假设 obs_data 的形状是 (n_episodes, max_seq_length, n_agents, obs_shape)
                state = obs_data[:, :, agent_id]  # (n_episodes, max_seq_length, obs_shape)
                state = state.reshape(-1, obs_shape)  # 展平为 (n_episodes * max_seq_length, obs_shape)

                # 创建对应的 label
                label = torch.full((state.shape[0],), role_list[agent_id])  # torch.full((n_episodes, max_seq_length), role_list[agent_id])
                # label = label.reshape(-1)  # 展平为 (n_episodes * max_seq_length,)
                states.append(state)
                labels.append(label)

                # state = obs_data[:, :, agent_id]  # 10 episodes, 100 timesteps, i agent, obs_shape
                # label = torch.full((len(exp_buffers[agent_id]),), role_list[agent_id])
                # 长度为buffer长度的tensor，每个元素都被填充为agent_id对应的角色label[attack, defence]
            
            # 将所有状态和标签连接成一个大的张量，比如 【(1000, 64) *4】变成 (4000, 64)
            states = torch.cat(states, dim=0)
            labels = torch.cat(labels, dim=0)
            assert states.shape[0] == labels.shape[0], "States and labels must have the same number of samples"

            dataset = SimpleListDataset(states, labels)
            train_size = int((1 - test_fraction) * len(dataset))
            test_size = len(dataset) - train_size
            train_dataset, test_dataset = torch.utils.data.random_split(dataset,
                                                                        [train_size, test_size])
            train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

        network_arch = [states[0].size().numel(), *hidden_sizes, 1]

        return cls(
            n_agents=n_agents,
            train_dataloader=train_dataloader, 
            test_dataloader=test_dataloader,
            network_arch=network_arch,
            role_list=role_list,
            learning_rate=learning_rate,
            batch_size=batch_size,
            test_period=test_period,
            obs_mask=obs_mask,
            )

    @classmethod
    def from_config_load(cls, n_agents, cfg, buffer_path):
        classifier = cls(
            n_agents,
            train_dataloader=None, 
            test_dataloader=None,
            network_arch=None, 
            role_list=None
        )
        # absolute_path = os.path.abspath(cfg.load_doe_name)
        absolute_path = os.path.join(buffer_path, cfg.load_doe_name)
        loaded_mlps = torch.load(absolute_path)
        
        # sanity check
        if not isinstance(loaded_mlps, list) or not all(isinstance(mlp, torch.nn.Module) for mlp in loaded_mlps):
            raise TypeError("Loaded object is not a list of torch.nn.Modules")
        
        classifier.mlps = loaded_mlps
        return classifier



    """ Merge 模式，返回一个空类，网络参数与cfg是一样的，只是cls.mlps里包含的agent个数不同，用于加载其他doe cls的参数 """
    # @classmethod
    # def from_merge_config(cls, n_agents, cfg):

    #     # 初始化role list，用-1代表没有分配角色
    #     role_list = [-1] * n_agents
    #     role_ids = cfg.get("role_ids")
    #     # (0, ('defence', ['alice', 'bob'])) 和 (1, ('attack', ['carol', 'dave']))
    #     # 这里role_ids是字典，key是角色，value是agent_id

    #     # 设置角色列表
    #     for label, (_, role_agents_ids) in enumerate(role_ids.items()):
    #         for agent_id in role_agents_ids:
    #             role_list[agent_id] = label
    #     # role_list = [0, 0, 1, 1, 1] 代表分别是 防御防御进攻进攻进攻，取决于任务num agents和yaml设置


    #     # 创建一个空的 MLPClassifier 实例
    #     return cls(
    #         n_agents=n_agents,
    #         train_dataloader=None,  # 不需要训练数据加载器
    #         test_dataloader=None,   # 不需要测试数据加载器
    #         network_arch=[0] * 3,   # 可以设置为默认的空网络架构，或根据需要调整
    #         role_list=role_list,     # 使用合并后的角色列表
    #         learning_rate=1e-2,      # 默认学习率
    #         batch_size=256,          # 默认批量大小
    #         test_period=5,           # 默认测试周期
    #         obs_mask=None             # 根据需要设置
    #     )




