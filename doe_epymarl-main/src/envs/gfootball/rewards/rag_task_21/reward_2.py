import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense reward based on defensive skills training, focusing on interceptions and player positions."""
    
    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.interceptions = 0
        self.defensive_positioning_reward = 0.05

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.interceptions = 0
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['interceptions'] = self.interceptions
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.interceptions = from_pickle['interceptions']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "interception_reward": [0.0] * len(reward),
                      "defensive_positioning_reward": [0.0] * len(reward)}
        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for i in range(len(reward)):
            o = observation[i]
            if 'ball_owned_team' in o and o['ball_owned_team'] == 0:  # Left team
                if 'ball_owned_player' in o and o['ball_owned_player'] == o['active']:
                    self.interceptions += 1
                    components['interception_reward'][i] = 1.0
                    reward[i] += 1.0 * components['interception_reward'][i]
            
            # Evaluate defensive positioning
            player_x, player_y = o['left_team'][o['active']]
            ball_x, ball_y = o['ball'][:2]

            if abs(player_x - ball_x) < 0.1 and abs(player_y - ball_y) < 0.1:
                components["defensive_positioning_reward"][i] = self.defensive_positioning_reward
                reward[i] += components["defensive_positioning_reward"][i]

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
                info[f"sticky_actions_{i}"] = self.sticky_actions_counter[i]
        return observation, reward, done, info
