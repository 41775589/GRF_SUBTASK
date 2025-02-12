import yaml

# # Load the YAML file
# task = 'Cartpole'
# suffix = 'GPT'

def create_task(root_dir, task, layer, response_id, response_r_id, num_agents, group_id, suffix):
    # Create task YAML file 
    input_file = f"{root_dir}/{task}.yaml"
    output_file = f"{root_dir}/{task}{suffix}_decomposition{response_id}_subtask{group_id}.yaml"

    with open(input_file, 'r') as yamlfile:
        data = yaml.safe_load(yamlfile)

    data["env_args"]["num_agents"] = num_agents
    data["env_args"]["map_name"] = f'scenario_layer{layer}_decomposition{response_id}_subtask{group_id}'
    data["env_args"]["rewards"] = f'scoring, reward_layer0_decomposition{response_id}_subtask{group_id}_sample{response_r_id}'
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