第一个是create task

```
def create_train_cfg(root_dir, Time, algs_name, layer, response_id, response_r_id, num_agents, group_id, iter, init_layer=False):
    """
    root_dir: dir - 读取/存储 alg config 路径  '/data/qiaodan/projects/GRF_SUBTASK/doe_epymarl-main/src/config/algs'
    Time: str - 指定的参数，如 0504
    algs_name: - ia2c
    layer: int - 2
    response_id: int - 0
    response_r_id: int - 奖励函数样本ID
    num_agents: int - 1
    group_id: int - 6
    iter: int - 0 GPT 迭代次数
    """
    # Create task YAML file
    input_file = f"{root_dir}/{algs_name}.yaml"
    output_file = f"{root_dir}/{algs_name}_layer{layer}_decomposition{response_id}_subtask{group_id}_iter{iter}_sample{response_r_id}.yaml"

    with open(input_file, 'r') as yamlfile:
        data = yaml.safe_load(yamlfile)

    # 添加基本训练参数（如果模版不存在）
    base_params = {
        # "hidden_dim": 128,
        "obs_agent_id": True, # 开启onehot
        "use_rnn": True,
        "save_buffer": True,
        "save_doe_cls": True
    }
    
    for key, value in base_params.items():
        data[key] = value

    data["layer_id"] = layer
    data["decomposition_id"] = response_id
    data["group_id"] = group_id
    data["iter_id"] = iter
    data["sample_id"] = response_r_id
    data["time_stamp"] = Time

    # 首层训练不用带doe
    # 这个不好指定，因为第一层初始化时，没有doe cls，load doe buffer和load doe name都是随便指定的，没有文件
    if not init_layer:
        data["use_doe"] = True  
        data['mac'] = "non_shared_doe_mac"  # 使用ns doe mac
    else:
        data["use_doe"] = False
        data['mac'] = "non_shared_mac"  # 使用ns mac

    

    # 测试加速debug用，正式训练需要删掉
    data["t_max"] = 2000
    data["batch_size_run"] = 1
    
    # 确保存在doe_classifier_cfg并设置其必需参数
    if "doe_classifier_cfg" not in data:
        data["doe_classifier_cfg"] = {}

    # train merge team函数会为merge team重新创建role_ids，这里更多是为最底层task创建role_ids. {"goal_6": [0, 1]}
    role_ids = {"goal_{}".format(group_id): [i for i in range(num_agents)]}

    # 设置DOE分类器配置
    doe_cfg = {
        "doe_type": "mlp",
        "load_mode": "train",
        "save_classifier": True,
        "save_doe_name": f"cls_layer{layer}_decomposition{response_id}_subtask{group_id}_iter{iter}_sample{response_r_id}.pt",
        "mlp": {
            "hidden_sizes": [128],
            "batch_size": 512,
            "test_fraction": 0.1,
            "learning_rate": 1e-2
        },
        "role_ids": role_ids  # 将在后面更新
    }

    # 更新DOE分类器配置，保留已有的配置
    data["doe_classifier_cfg"].update(doe_cfg)

    # 本层实验的所有存储文件统一文件夹
    layer_data_save_dir=f'~/projects/GRF_SUBTASK/doe_epymarl-main/results/gfootball/{Time}/decomposition{response_id}/group{group_id}'
    layer_data_save_dir = os.path.expanduser(layer_data_save_dir)
    data["doe_classifier_cfg"]["layer_tmp_dir"] = layer_data_save_dir

    # Write the new YAML file
    with open(output_file, 'w') as new_yamlfile:
        yaml.safe_dump(data, new_yamlfile)
```


第二个是doe ia2c learner中load models

```
def load_models(self, path):
        #已在doe_controller改为actor_init
        # 加载合并后的actor
        actor_state_dict = th.load("{}/actor_init.th".format(path), map_location=lambda storage, loc: storage)
        # 由于mac.agents是ModuleList包含2个RNNAgent，需要分别加载
        for agent_id, agent in enumerate(self.mac.agent.agents):
            # 创建新的state dict，移除"agents.0."前缀
            new_state_dict = {}
            for k, v in actor_state_dict[agent_id].items():
                # 方法1：使用split
                new_k = k.split('.')
                if new_k[0] == 'agents':  # 确保是以agents开头
                    new_k = '.'.join(new_k[2:])  # 跳过'agents.X'，直接取后面的部分
                else:
                    new_k = k  # 如果不是以agents开头，保持原样
                new_state_dict[new_k] = v
            # 加载处理后的state dict
            agent.load_state_dict(new_state_dict)
        
        # 加载合并后的critic
        critic_state_dict = th.load("{}/critic_init.th".format(path), map_location=lambda storage, loc: storage)
        # critic.critics是list包含2个critic，需要分别加载
        for critic_id, critic in enumerate(self.critic.critics):
            critic.load_state_dict(critic_state_dict[critic_id])
        
        # 同步target critic
        self.target_critic.load_state_dict(self.critic.state_dict())


        # self.mac.load_models(path)
        # # 加载 critic 和 target_critic 的权重
        # self.critic.load_state_dict(th.load("{}/critic_init.th".format(path), map_location=lambda storage, loc: storage))
        # self.target_critic.load_state_dict(self.critic.state_dict())
        # # 注意：这里不加载 optimizer，因此不支持中断训练后继续
```