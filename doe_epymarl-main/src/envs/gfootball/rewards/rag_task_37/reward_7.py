import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that adds advanced ball control and passing rewards
    under pressure in tight game situations.
    """

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)

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
                      "ball_control_reward": [0.0] * len(reward)}

        # Check for pressure situation and reward passes
        for rew_index in range(len(reward)):
            o = observation[rew_index]
            if o['ball_owned_team'] == 0:  # When the left team has the ball
                opponent_distances = np.sqrt(np.sum((o['left_team'][o['active']] - o['right_team'])**2, axis=1))
                close_opponents = np.sum(opponent_distances < 0.1)  # Number of opponents within a range considered as under pressure
                
                if close_opponents > 2:  # Consider it a high-pressure situation only if more than two opponents are close
                    pass_type_reward = 0.0
                    if o['sticky_actions'][6]:  # action_high_pass
                        pass_type_reward = 0.3
                    elif o['sticky_actions'][5]:  # action_long_pass
                        pass_type_reward = 0.2
                    elif o['sticky_actions'][4]:  # action_short_pass
                        pass_type_reward = 0.1
                    
                    components['ball_control_reward'][rew_index] = pass_type_reward
            
            # Sum up the rewards for the final decision
            reward[rew_index] += components['ball_control_reward'][rew_index]

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
                if action:
                    self.sticky_actions_counter[i] += 1
                info[f"sticky_actions_{i}"] = self.sticky_actions_counter[i]
        return observation, reward, done, info
