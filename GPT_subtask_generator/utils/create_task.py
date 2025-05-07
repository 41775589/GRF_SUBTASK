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


def create_train_cfg(root_dir, Time, algs_name, layer, response_id, response_r_id, num_agents, group_id, iter):
    """
    root_dir: dir - 读取/存储 alg config 路径  '/data/qiaodan/projects/GRF_SUBTASK/doe_epymarl-main/src/config/algs'
    Time: str - 指定的参数，如 0504
    algs_name: - ia2c
    layer: int - 2
    response_id: int - 0
    num_agents: int - 1
    group_id: int - 6
    iter: int - 0
    """
    # Create task YAML file
    input_file = f"{root_dir}/{algs_name}.yaml"
    output_file = f"{root_dir}/{algs_name}_layer{layer}_decomposition{response_id}_subtask{group_id}_iter{iter}_sample{response_r_id}.yaml"

    with open(input_file, 'r') as yamlfile:
        data = yaml.safe_load(yamlfile)

    data["layer_id"] = layer
    data["decomposition_id"] = response_id
    data["group_id"] = group_id
    data["iter_id"] = iter
    data["sample_id"] = response_r_id
    data["time_stamp"] = Time

    data["doe_classifier_cfg"]["role_ids"]={"task":[]}
    for i in range(num_agents):
        data["doe_classifier_cfg"]["role_ids"]['task'].append(i)

    data["doe_classifier_cfg"]["save_doe_name"] = f"cls_layer{layer}_decomposition{response_id}_subtask{group_id}_iter{iter}_sample{response_r_id}.pt"

    # 新增：本层实验的所有存储文件统一文件夹
    layer_data_save_dir=f'~/zihao/PycharmProjects/GRF_SUBTASK-tmp/doe_epymarl-main/results/gfootball/{Time}/decomposition{response_id}/group{group_id}'
    layer_data_save_dir = os.path.expanduser(layer_data_save_dir)
    data["doe_classifier_cfg"]["layer_tmp_dir"] = layer_data_save_dir


    # TODO 这个函数只在最底层使用，不需要load policy，修改写在train_merge_team里
    # load doe name
    # load doe buffer path
    # ckpt path 不要为空（目前是从default加载为空）



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