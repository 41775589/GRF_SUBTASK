import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that rewards agents for precise close-range attacks and effective dribbling near the goal."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.close_range_precision_bonus = 0.5
        self.dribbling_effectiveness_bonus = 0.3

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        state_pickle = self.env.get_state(to_pickle)
        state_pickle['CheckpointRewardWrapper'] = {
            'sticky_actions_count': self.sticky_actions_counter.tolist()
        }
        return state_pickle

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        sticky_actions_count = from_pickle.get('CheckpointRewardWrapper', {}).get('sticky_actions_count', None)
        if sticky_actions_count is not None:
            self.sticky_actions_counter = np.array(sticky_actions_count, dtype=int)
        return from_pickle

    def reward(self, reward):
        observations = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy()}

        for i, r in enumerate(reward):
            o = observations[i]
            if o['game_mode'] == 0 and o['active'] > -1:  # Normal game mode and active player
                # Check how close the agent is to the opposition goal when shooting
                x_pos = o['right_team'][o['active']][0]
                if x_pos > 0.8 and o['ball_owned_team'] == 1:  # Close to the opposition's goal
                    reward[i] += self.close_range_precision_bonus

                # Check dribbling actions
                if o['sticky_actions'][9] == 1:  # If dribbling
                    reward[i] += self.dribbling_effectiveness_bonus

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
