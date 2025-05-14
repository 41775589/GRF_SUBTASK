import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a reward for passive defensive techniques focused on stopping movement and sprinting."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        # Define parameters to adjust the significance of the rewards
        self.stop_movement_reward = 0.05
        self.stop_sprint_reward = 0.03

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper_sticky_actions'] = self.sticky_actions_counter
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle.get('CheckpointRewardWrapper_sticky_actions', np.zeros(10, dtype=int))
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        base_score_reward = reward.copy()
        passive_defense_reward = [0.0] * len(reward)
        
        if observation is None:
            return reward, {'base_score_reward': base_score_reward, 'passive_defense_reward': passive_defense_reward}

        for idx in range(len(reward)):
            o = observation[idx]
            
            if o['sticky_actions'][7] == 1:  # Assuming index 7 refers to 'Stop Moving'
                # Assigning positional importance to defensive stopping
                if np.linalg.norm(o['right_team'][o['active']] - [1, 0]) > 0.8:
                    passive_defense_reward[idx] += self.stop_movement_reward

            if o['sticky_actions'][8] == 1:  # Assuming index 8 refers to 'Stop Sprint'
                # Encourage the use of 'Stop Sprint' intelligently depending on game state
                positions = o['right_team'] if o['ball_owned_team'] == 1 else o['left_team']
                average_field_position = np.mean(positions[:, 0])
                if (-1 <= average_field_position <= 1):
                    passive_defense_reward[idx] += self.stop_sprint_reward

            # Aggregate rewards
            reward[idx] += passive_defense_reward[idx]

        return reward, {'base_score_reward': base_score_reward, 'passive_defense_reward': passive_defense_reward}

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            if agent_obs:
                for i, action in enumerate(agent_obs['sticky_actions']):
                    self.sticky_actions_counter[i] += action
        return observation, reward, done, info
