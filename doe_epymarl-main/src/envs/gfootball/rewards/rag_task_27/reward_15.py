import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense checkpoint reward based on defensive responsiveness and interception skills."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        # Define multiple checkpoints based on strategic positions for defense
        self._interceptions = {}

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self._interceptions = {}
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = self._interceptions
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self._interceptions = from_pickle['CheckpointRewardWrapper']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "interception_reward": [0.0] * len(reward)}
        if observation is None:
            return reward, components
        
        for rew_index in range(len(reward)):
            o = observation[rew_index]
            
            if 'ball_owned_team' in o and o['ball_owned_team'] == 1 and o['game_mode'] == 0:
                # Calculate the distance to the ball
                player_pos = o['left_team'][o['active']]
                ball_pos = o['ball'][0:2]  # ignoring z coordinate
                distance_to_ball = np.linalg.norm(player_pos - ball_pos)
                
                # Reward interceptions: predictions or actions that prevent the opponent from advancing
                if distance_to_ball < 0.03 and not self._interceptions.get(rew_index, False):
                    components['interception_reward'][rew_index] = 0.5
                    reward[rew_index] += components['interception_reward'][rew_index]
                    self._interceptions[rew_index] = True

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info['final_reward'] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        self.sticky_actions_counter.fill(0)
        for agent_obs in observation:
            for i, action in enumerate(agent_obs['sticky_actions']):
                info[f"sticky_actions_{i}"] = action
        return observation, reward, done, info
