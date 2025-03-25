import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense reward for mastering long passes in football game."""

    def __init__(self, env):
        super().__init__(env)
        self.pass_initiation = {}
        self.reward_scale = 0.05
        self.pass_length_threshold = 0.3  # Arbitrarily selected threshold for long pass
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.pass_initiation = {}
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = self.pass_initiation
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.pass_initiation = from_pickle['CheckpointRewardWrapper']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(), "long_pass_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            player_pos = o['left_team'][o['active']] if o['ball_owned_team'] == 0 else o['right_team'][o['active']]
            
            # Start tracking a new significant ball movement (long pass initiation)
            if (o['ball_owned_player'] == o['active'] and
               o['ball_owned_team'] in [0, 1]):
                self.pass_initiation[rew_index] = player_pos
            
            # Evaluate if the ball was passed long distance before being caught or goes out of reception
            if (o['ball_owned_player'] != o['active'] or o['ball_owned_team'] == -1) and rew_index in self.pass_initiation:
                pass_origin = self.pass_initiation.pop(rew_index)
                current_ball_pos = o['ball'][:2]
                dist_travelled = np.linalg.norm(np.array(pass_origin) - np.array(current_ball_pos))

                if dist_travelled > self.pass_length_threshold:
                    components['long_pass_reward'][rew_index] = self.reward_scale * dist_travelled
                    reward[rew_index] += components['long_pass_reward'][rew_index]

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        obs = self.env.unwrapped.observation()
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                info[f"sticky_actions_{i}"] = action
        return observation, reward, done, info
