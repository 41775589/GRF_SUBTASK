import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that adds a dense reward based on ball passing under pressure,
    aiming to train agents to master short passing and retention under defensive pressure.
    """
    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.num_pass_zones = 5
        self.pass_reward = 0.2  # Added reward for passing in each defined zone

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        # no specific state to update here, this is a placeholder
        return from_pickle

    def calc_distance(self, pos1, pos2):
        """Calculates Euclidean distance between two 2D positions."""
        return np.sqrt((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2)

    def reward(self, reward):
        components = {"base_score_reward": reward.copy(), "passing_reward": [0.0, 0.0]}
        observation = self.env.unwrapped.observation()
        
        if observation is None:
            return reward, components
        
        assert len(reward) == len(observation)

        for i in range(len(reward)):
            obs = observation[i]
            if 'ball_owned_team' in obs and obs['ball_owned_team'] == 0:  # possession by left team
                player_pos = obs['left_team'][obs['ball_owned_player']]
                opponents = obs['right_team']
                
                close_opponents = np.sum([self.calc_distance(player_pos, opp) < 0.1 for opp in opponents])
                if close_opponents > 0:
                    components["passing_reward"][i] = self.pass_reward * min(1, close_opponents / 3)
                    reward[i] += components["passing_reward"][i]

        return reward, components

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        current_observation = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in current_observation:
            for idx, act in enumerate(agent_obs['sticky_actions']):
                info[f"sticky_actions_{idx}"] = act
        return obs, reward, done, info
