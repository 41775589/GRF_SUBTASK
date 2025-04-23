import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a reward for successful long-range shots and controlling the ball in the opponent's half."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.long_shot_threshold = 0.7  # X coordinate threshold for a "long-range" shot

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
        components = {'base_score_reward': reward.copy(),
                      'long_shot_reward': [0.0] * len(reward)}
        
        if observation is None:
            return reward, components

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            if o['game_mode'] != 0:  # Not in normal play mode
                continue

            # Check if the right team has possession and is close to making a shot
            if (o['ball_owned_team'] == 1 and
                o['ball'][0] > self.long_shot_threshold and
                'ball_owned_player' in o and
                o['ball_owned_player'] == o['designated']):  # The designated player is in control of the ball

                # Check if shot is occurring
                if o['right_team'][o['designated']][0] > self.long_shot_threshold:
                    components['long_shot_reward'][rew_index] = 1.0
                    reward[rew_index] += components['long_shot_reward'][rew_index]
        
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
