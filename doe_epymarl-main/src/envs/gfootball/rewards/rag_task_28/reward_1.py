import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense reward for dribbling skills against a goalkeeper."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.dribble_reward_coefficient = 0.5
        self.direction_change_reward_coefficient = 0.3
        self.ball_control_duration_reward = 0.2

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        state = self.env.get_state(to_pickle)
        state['sticky_actions_counter'] = self.sticky_actions_counter.tolist()
        return state

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = np.array(from_pickle['sticky_actions_counter'])
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "dribble_reward": [0.0] * len(reward),
                      "direction_change_reward": [0.0] * len(reward),
                      "ball_control_duration_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            
            if o['ball_owned_team'] == 1 and o['ball_owned_player'] == o['active']:
                # Reward dribbling (using dribble action)
                if o['sticky_actions'][-1] == 1:  # Assuming dribble action index is the last
                    components["dribble_reward"][rew_index] += self.dribble_reward_coefficient
                    reward[rew_index] += components["dribble_reward"][rew_index]
                
                # Reward changes in direction while dribbling
                direction_changes = np.sum(np.abs(np.diff(o['right_team_direction'][o['active']])))
                components["direction_change_reward"][rew_index] = direction_changes * self.direction_change_reward_coefficient
                reward[rew_index] += components["direction_change_reward"][rew_index]
                
                # Reward maintaining ball control
                self.sticky_actions_counter[o['active']] += 1
                components["ball_control_duration_reward"][rew_index] = self.ball_control_duration_reward * self.sticky_actions_counter[o['active']]
                reward[rew_index] += components["ball_control_duration_reward"][rew_index]
            
            else:
                # Reset the counter for ball control if not in control
                self.sticky_actions_counter[o['active']] = 0

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                info[f"sticky_actions_{i}"] = action
        return observation, reward, done, info
