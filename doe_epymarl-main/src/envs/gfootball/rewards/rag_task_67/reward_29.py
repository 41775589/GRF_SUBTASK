import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A reward wrapper to focus on skills aiding transition from defense to attack,
    such as Short Pass, Long Pass, and Dribble, achieving ball control under pressure.
    """
    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.passing_bonus = 0.2
        self.dribbling_bonus = 0.1

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['custom_state_key'] = self.sticky_actions_counter
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['custom_state_key']
        return from_pickle

    def reward(self, reward):
        # Init the components tracking dictionary
        components = {
            "base_score_reward": reward.copy(),
            "passing_bonus": [0.0] * len(reward),
            "dribbling_bonus": [0.0] * len(reward)
        }

        observation = self.env.unwrapped.observation()
        if observation is None:
            return reward, components

        for i, o in enumerate(observation):
            # Reward for successful passes or dribbles under pressure
            if o['game_mode'] in {2, 3, 4, 5}:  # These modes involve set pieces or throw-ins
                if o['ball_owned_team'] == o['active']:
                    components['passing_bonus'][i] = self.passing_bonus
                    reward[i] += components['passing_bonus'][i]

            # In normal play, reward dribble holding ball under pressure
            if o['game_mode'] == 0:  # Normal gameplay
                if o['sticky_actions'][9] == 1:  # dribbling action is activated
                    components['dribbling_bonus'][i] = self.dribbling_bonus
                    reward[i] += components['dribbling_bonus'][i]

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
