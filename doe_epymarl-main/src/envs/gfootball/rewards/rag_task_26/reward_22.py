import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a midfield mastery reward, targeting both defensive and attacking contributions."""

    def __init__(self, env):
        super().__init__(env)
        self.num_midfielders = 4  # Assuming 4 midfielders in control (could be different based on setup)
        self.midfield_control_rewards = np.zeros(self.num_midfielders)
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.midfield_control_rewards.fill(0)
        self.sticky_actions_counter.fill(0)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['midfield_control_rewards'] = self.midfield_control_rewards
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.midfield_control_rewards = from_pickle['midfield_control_rewards']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "midfield_control_reward": np.zeros(len(reward))}

        for i, o in enumerate(observation):
            midfield_control = 0
            if ('left_team_roles' in o and 'right_team_roles' in o):
                midfield_roles = np.where(np.isin(o['left_team_roles'], [4, 5, 6, 7]))[0]
                
                # Reward midfielders for maintaining ball possession or retrieving it
                if o['ball_owned_team'] == 0 and o['ball_owned_player'] in midfield_roles:
                    midfield_control += 0.1  # Reward for ball control by midfielders

                # Assist or contribution to goal, this may be tracked via events or complex state 
                # analysis in an actual implementation
                if o['score'][0] > 0:
                    previous_score = components["base_score_reward"][i]  # Assuming access to previous score
                    current_score = o['score'][0]
                    if current_score > previous_score:
                        midfield_control += 0.5  # Significant reward for contributing to a score

            components["midfield_control_reward"][i] += midfield_control
            reward[i] += components["midfield_control_reward"][i]

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
