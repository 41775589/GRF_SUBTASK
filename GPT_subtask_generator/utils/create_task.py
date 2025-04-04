import yaml

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
    # data["env_args"]["rewards"] = 'scoring, reward_test'

    # data["t_max"] = 200
    
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