import yaml
import os

# # Load the YAML file
# task = 'Cartpole'
# suffix = 'GPT'

def create_task(root_dir, task, layer, response_id, response_r_id, num_agents, group_id, iter, suffix):
    # Create task YAML file 
    input_file = f"{root_dir}/{task}.yaml"
    output_file = f"{root_dir}/{task}{suffix}_layer{layer}_decomposition{response_id}_subtask{group_id}_iter{iter}_sample{response_r_id}.yaml"

    with open(input_file, 'r') as yamlfile:
        data = yaml.safe_load(yamlfile)

    data["env_args"]["num_agents"] = num_agents
    data["env_args"]["map_name"] = f'scenario_layer{layer}_decomposition{response_id}_subtask{group_id}'
    data["env_args"]["rewards"] = f'scoring, reward_layer{layer}_decomposition{response_id}_subtask{group_id}_iter{iter}_sample{response_r_id}'


    # # 这两行正常是注释掉的，这里为了测试test加回来
    # data["env_args"]["rewards"] = 'scoring, reward_test'
    # data["t_max"] = 2000
    
    # Write the new YAML file
    with open(output_file, 'w') as new_yamlfile:
        yaml.safe_dump(data, new_yamlfile)

    # We don't need this part, algs yaml files are pre-defined

    # # Create training YAML file
    # input_file = f"{root_dir}/cfg/train/{task}PPO.yaml"
    # output_file = f"{root_dir}/cfg/train/{task}{suffix}PPO.yaml"
    #
    # with open(input_file, 'r') as yamlfile:
    #     data = yaml.safe_load(yamlfile)
    #
    # # Modify the "name" field
    # data['params']['config']['name'] = data['params']['config']['name'].replace(task, f'{task}{suffix}')
    #
    # # Write the new YAML file
    # with open(output_file, 'w') as new_yamlfile:
    #     yaml.safe_dump(data, new_yamlfile)


def create_train_cfg(root_dir, Time, algs_name, layer, response_id, response_r_id, num_agents, group_id, iter, task_env, results_dir, init_layer=False):
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

    

    # # 测试加速debug用，正式训练需要删掉
    # data["t_max"] = 2000
    # data["batch_size_run"] = 1
    
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
    data["doe_classifier_cfg"][
        "layer_tmp_dir"] = f"{results_dir}/{task_env}/{Time}/decomposition{response_id}/group{group_id}"

    # Write the new YAML file
    with open(output_file, 'w') as new_yamlfile:
        yaml.safe_dump(data, new_yamlfile)


def create_task_RAG(root_dir, task, num_agents, reward_id, map_name):
    # Create task YAML file
    input_file = f"{root_dir}/{task}.yaml"
    output_file = f"{root_dir}/{task}_RAG_reward{reward_id}.yaml"

    with open(input_file, 'r') as yamlfile:
        data = yaml.safe_load(yamlfile)

    data["env_args"]["num_agents"] = num_agents
    data["env_args"]["rewards"] = f'scoring, reward_{reward_id}'
    data["env_args"]["map_name"] = map_name


    # Write the new YAML file
    with open(output_file, 'w') as new_yamlfile:
        yaml.safe_dump(data, new_yamlfile)

def create_train_cfg_RAG(root_dir, Time, algs_name, reward_id, num_agents):
    # Create task YAML file
    input_file = f"{root_dir}/{algs_name}.yaml"
    output_file = f"{root_dir}/{algs_name}_RAG_reward{reward_id}.yaml"

    with open(input_file, 'r') as yamlfile:
        data = yaml.safe_load(yamlfile)

    data["layer_id"] = reward_id
    data["decomposition_id"] = 0
    data["group_id"] = 0
    data["iter_id"] = 0
    data["sample_id"] = 0
    data["time_stamp"] = Time

    data["doe_classifier_cfg"]["role_ids"]={"task":[]}
    for i in range(num_agents):
        data["doe_classifier_cfg"]["role_ids"]['task'].append(i)

    data["doe_classifier_cfg"]["save_doe_name"] = f"cls_0.pt"


    # Write the new YAML file
    with open(output_file, 'w') as new_yamlfile:
        yaml.safe_dump(data, new_yamlfile)