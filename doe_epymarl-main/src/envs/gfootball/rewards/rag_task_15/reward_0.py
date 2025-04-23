import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A reward wrapper to focus on mastering long passes in a football game, emphasizing accuracy 
    and understanding of ball dynamics over long distances.
    """

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.pass_accuracy_threshold = 0.5  # Minimum distance for rewarding a pass attempt
        self.pass_accuracy_reward = 0.2      # Reward for a successful long pass

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = self.sticky_actions_counter
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['CheckpointRewardWrapper']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "pass_accuracy_reward": [0.0] * len(reward)}
        
        if observation is None:
            return reward, components
        
        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]

            if 'ball_owned_team' in o and o['ball_owned_team'] == 0:  # Check if left team owns the ball
                ball_destination = np.array(o['ball']) + np.array(o['ball_direction']) * 10  # Predictive destination
                distance = np.sqrt(np.sum(np.power(np.array(o['ball']) - ball_destination, 2)))

                if distance >= self.pass_accuracy_threshold:
                    components["pass_accuracy_reward"][rew_index] = self.pass_accuracy_reward
                    reward[rew_index] += components["pass_accuracy_reward"][rew_index] * (distance - self.pass_accuracy_threshold)

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
                self.sticky_actions_counter[i] += action
                info[f"sticky_actions_{i}"] = action
        return observation, reward, done, info
