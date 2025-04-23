import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper to reinforce dribbling skills when facing the goalkeeper.
    It encourages quick feints, sudden direction changes, and consistent ball control under pressure.
    """
    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self._checkpoint_reward = 0.5
        # Dribbling reward when nearing the goalkeeper
        self.goalkeeper_proximity_reward = 1.0

    def reset(self):
        """
        Reset the wrapper heuristics at the start of an episode.
        """
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        """
        Save the state of this wrapper along with the environment's state.
        """
        to_pickle['sticky_actions_counter'] = self.sticky_actions_counter
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """
        Restore the state of the wrapper from given state information.
        """
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle.get('sticky_actions_counter', np.zeros(10, dtype=int))
        return from_pickle

    def reward(self, reward):
        """
        Reward function focusing on dribbling towards and around the goalkeeper.
        Attributes enhanced rewards based on proximity to goalkeeper and actions taken.
        """
        observation = self.env.unwrapped.observation()
        components = {'base_score_reward': reward.copy(), 'dribbling_reward': [0.0] * len(reward)}

        if observation is None:
            return reward, components

        for rew_index, o in enumerate(observation):
            # Adding reward if the player is dribbling close to the goalkeeper
            goalkeeper_pos = o['right_team'][0]  # Assuming index 0 is the goalkeeper
            player_pos = o['left_team'][o['active']]
            dist_to_goalkeeper = np.linalg.norm(player_pos - goalkeeper_pos)

            # Encourage dribbling when close to the goalkeeper (within a distance threshold)
            if dist_to_goalkeeper < 0.15:
                components['dribbling_reward'][rew_index] = self.goalkeeper_proximity_reward
                if 'sticky_actions' in o and o['sticky_actions'][9]:  # Action 9 - dribbling
                    components['dribbling_reward'][rew_index] += self._checkpoint_reward

            # Update the reward
            reward[rew_index] += components['dribbling_reward'][rew_index]

        return reward, components

    def step(self, action):
        """
        Take a step using the underlying env, apply the reward wrapper logic, and return modified outputs.
        """
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        
        # Collect sticky actions data to analyze agent's behavior patterns.
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                self.sticky_actions_counter[i] += action

        return observation, reward, done, info
