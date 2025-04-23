import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A custom gym wrapper that adjusts rewards specifically for goalkeeper training
    including shot-stopping, reflex enhancement, and initiating counter-attacks with accurate passing.
    """

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int) # Tracking sticky actions utilized by agents

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = self.sticky_actions_counter.tolist()
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = np.array(from_pickle['CheckpointRewardWrapper'])
        return from_pickle

    def reward(self, reward):
        """
        Calculate the diversity of various rewards:
        - Reward stopping shots
        - Reward quick reflex responses
        - Reward effective passes that initiate counter-attacks
        """
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(), "save_reward": [0.0, 0.0], "quick_reflex": [0.0, 0.0], "efficient_pass": [0.0, 0.0]}

        if observation is None:
            return reward, components

        # Define coefficients for different components
        save_coefficient = 1.0
        reflex_coefficient = 0.5
        pass_coefficient = 0.5

        for rew_index in range(len(reward)):
            o = observation[rew_index]

            # Extract necessary elements from observation
            goalkeeper_role = 0
            has_ball = o['active'] if 'ball_owned_player' in o and o['ball_owned_team'] == o['left_team_roles'][o['active']] == goalkeeper_role else -1

            # Rewards for shot stopping
            if 'game_mode' in o and o['game_mode'] == 6: # Assuming game_mode 6 is a penalty or a close shot situation
                if reward[rew_index] == 0:  # No goal conceded
                    reward[rew_index] += save_coefficient
                    components["save_reward"][rew_index] += save_coefficient

            # Quick reflex reward - detected by sudden ball direction change near the goal area
            if 'ball_direction' in o and np.linalg.norm(o['ball_direction'][:2]) > 0.1:
                if abs(o['ball'][0]) > 0.9 and o['left_team'][goalkeeper_role][0] > 0.9: # close to the goal
                    reward[rew_index] += reflex_coefficient
                    components["quick_reflex"][rew_index] += reflex_coefficient

            # Efficient pass - Starting a counter attack by passing ball towards team-mate advancing forward
            if 'right_team_direction' in o and has_ball != -1:
                for i, direction in enumerate(o['right_team_direction']):
                    if direction[0] > 0 and np.linalg.norm(direction) > 0.25: # team member moving forward
                        reward[rew_index] += pass_coefficient
                        components["efficient_pass"][rew_index] += pass_coefficient

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
