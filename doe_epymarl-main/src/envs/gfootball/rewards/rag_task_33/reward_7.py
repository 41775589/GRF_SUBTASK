import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that adds a reward for long-range shooting, specialized to encourage shots from outside the penalty box.
    This includes beating opposing defenders and making decision to shoot from distance.
    """

    def __init__(self, env):
        super().__init__(env)
        # Define threshold for long range, typically outside the penalty box (peanlty box ~16m from goal line = ~0.27 normalized x coordinate)
        self.long_range_threshold = 0.27
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        # Define additional rewards
        self.long_shot_reward = 0.5  # additional reward for attempting long-range shots
        self.score_reward = 1.0      # additional reward for scoring

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        return self.env.set_state(state)

    def reward(self, reward):
        """
        Enhances the base reward by adding specific rewards for long-range shots,
        especially those that lead to goals.
        """
        observation = self.env.unwrapped.observation()
        components = {'base_score_reward': reward.copy(),
                      'long_shot_reward': [0.0] * len(reward),
                      'score_reward': [0.0] * len(reward)}
        
        if observation is None:
            return reward, components

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            ball_position_x = o['ball'][0]
            if o['ball_owned_team'] == 1 and o['ball_owned_player'] == o['active']:
                # Check if the shot is taken from long range
                if ball_position_x > -self.long_range_threshold and ball_position_x < self.long_range_threshold:
                    components['long_shot_reward'][rew_index] = self.long_shot_reward
                    reward[rew_index] += components['long_shot_reward'][rew_index]

            if 'score' in o and o['score'][1] - o['score'][0] > 0:  # assuming control team is right team
                components['score_reward'][rew_index] = self.score_reward
                reward[rew_index] += components['score_reward'][rew_index]

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
