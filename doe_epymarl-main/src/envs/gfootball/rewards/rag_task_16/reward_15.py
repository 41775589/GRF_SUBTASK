import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that adds a dense reward focused on enhancing high pass skills.
    Rewards are given based on the accuracy and strategic deployment of high passes.
    """

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        
        # Reward details
        self.pass_accuracy_threshold = 0.2  # threshold for assuming pass is accurate
        self.high_pass_reward = 0.5         # reward for achieving a high pass
        self.controlled_reception_reward = 0.3  # bonus when pass is received in control

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "high_pass_reward": [0.0] * len(reward),
                      "controlled_reception_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components
        
        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            
            # Check if a high pass was executed
            if o['ball_direction'][2] > self.pass_accuracy_threshold:  # Assuming third coordinate is the vertical component
                components["high_pass_reward"][rew_index] = self.high_pass_reward
                reward[rew_index] += self.high_pass_reward
                
                # Check if the pass is received by a teammate in a strategic position
                if (o['ball_owned_team'] == 1 and o['ball'][0] > 0.5):  # Assuming opponent team's half starts from 0.5 on x-axis
                    components["controlled_reception_reward"][rew_index] += self.controlled_reception_reward
                    reward[rew_index] += self.controlled_reception_reward

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
            for i, action_type in enumerate(agent_obs['sticky_actions']):
                self.sticky_actions_counter[i] = action_type
                info[f"sticky_actions_{i}"] = action_type
        return observation, reward, done, info
