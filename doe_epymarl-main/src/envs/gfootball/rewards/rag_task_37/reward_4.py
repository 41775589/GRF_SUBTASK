import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that adds a dense reward focusing on ball control and accurate passing
    under pressure, prioritizing short, high, and long passes in tight situations.
    """
    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {
            'sticky_actions_counter': self.sticky_actions_counter
        }
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['CheckpointRewardWrapper']['sticky_actions_counter']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        # Initialize reward components for each agent
        components = {"base_score_reward": reward.copy(),
                      "control_and_passing_reward": [0.0] * len(reward)}
        
        if observation is None:
            return reward, components

        # Ensure reward array length matches observation length
        assert len(reward) == len(observation)

        # Define multipliers for different passes under pressure
        short_pass_multiplier = 0.2
        high_pass_multiplier = 0.3
        long_pass_multiplier = 0.5

        for rew_index, o in enumerate(observation):
            # Base reward
            base_score = reward[rew_index]

            # Game situation analysis: ball controlled by the player
            if o['ball_owned_team'] == 0 and o['ball_owned_player'] == o['active']:
                # Assess the proximity of any opposing team members
                distance_threshold = 0.2  # threshold to define "pressure"
                close_opponents = np.any([
                    np.linalg.norm(o['right_team'][i] - o['left_team'][o['active']], ord=2) < distance_threshold
                    for i in range(len(o['right_team']))
                ])
                
                # Reward based on pass type and game situation
                if close_opponents:
                    ball_control_action = o['sticky_actions'][7:10]  # 7: short pass, 8: high pass, 9: long pass

                    if ball_control_action[0]:  # Short pass under pressure
                        components["control_and_passing_reward"][rew_index] += short_pass_multiplier
                    elif ball_control_action[1]:  # High pass under pressure
                        components["control_and_passing_reward"][rew_index] += high_pass_multiplier
                    elif ball_control_action[2]:  # Long pass under pressure
                        components["control_and_passing_reward"][rew_index] += long_pass_multiplier

            # Update total reward with new components
            reward[rew_index] = base_score + components["control_and_passing_reward"][rew_index]

        return reward, components
    
    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        # Update sticky_action tracking
        self.sticky_actions_counter.fill(0)
        for agent_obs in observation:
            for i, action in enumerate(agent_obs['sticky_actions']):
                self.sticky_actions_counter[i] += action
        return observation, reward, done, info
