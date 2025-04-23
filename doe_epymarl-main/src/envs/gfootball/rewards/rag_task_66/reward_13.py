import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense reward based on short passes under defensive pressure."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.num_passes = 5
        self.pass_reward = 0.2

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
                      "pass_reward": [0.0] * len(reward)}
        
        if observation is None:
            return reward, components

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            # Check if the pass has occurred under defensive pressure
            if ('ball_owned_player' in o and o['ball_owned_player'] == o['active'] and
                o['ball_owned_team'] == 0 and 'right_team_active' in o):
                # close proximity of defenders to the ball handler
                player_pos = o['left_team'][o['ball_owned_player']]
                defenders = o['right_team']
                dists = [np.linalg.norm(player_pos - d_pos) for d_pos in defenders]
                pressure = sum(d <= 0.1 for d in dists)  # arbitrary defensive pressure threshold
                
                if pressure > 0:
                    components["pass_reward"][rew_index] = self.pass_reward
                    reward[rew_index] += self.pass_reward

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
