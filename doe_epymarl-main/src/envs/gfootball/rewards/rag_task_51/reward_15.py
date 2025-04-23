import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that adds a dense reward for specialized goalkeeper training,
    focusing on shot-stopping, quick reflexes, and initiating counter-attacks.
    """

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.goalkeeper_actions = {'shot_stopping': 0.3, 'quick_reflexes': 0.5, 'initiate_counter': 0.2}

    def reset(self):
        """
        Reset the environment and sticky actions counter.
        """
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        """
        Save the modifications in state representation when pickling.
        """
        to_pickle['sticky_actions_counter'] = self.sticky_actions_counter
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """
        Restore the modifications in state representation when unpickling.
        """
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['sticky_actions_counter']
        return from_pickle

    def reward(self, reward):
        """
        Add specialized rewards related to goalkeeper behaviors.
        """
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "shot_stopping": [0.0, 0.0],
                      "quick_reflexes": [0.0, 0.0],
                      "initiate_counter": [0.0, 0.0]}

        for rew_index in range(len(reward)):
            o = observation[rew_index]

            # Identify the goalkeeper and determine if involved in a play
            if o['left_team_roles'][o['active']] == 0:  # assuming 0 is the role index for goalkeepers
                role = 'goalkeeper'
            else:
                continue

            # Additional rewards for negating shots (shot stopping)
            if role == 'goalkeeper' and o['game_mode'] in (3, 6) and o['ball_owned_team'] == 0:
                components['shot_stopping'][rew_index] = self.goalkeeper_actions['shot_stopping']
                reward[rew_index] += components['shot_stopping'][rew_index]

            # Rewards for quick reflex actions
            if role == 'goalkeeper' and np.linalg.norm(o['ball_direction']) > 0.2:
                components['quick_reflexes'][rew_index] = self.goalkeeper_actions['quick_reflexes']
                reward[rew_index] += components['quick_reflexes'][rew_index]

            # Rewards for initiating counter-attacks
            if role == 'goalkeeper' and o['ball_owned_team'] == 0 and o['active'] == o['ball_owned_player']:
                if 'right_team_direction' in o and np.any(o['right_team_direction'][o['active']] > 0.1):
                    components['initiate_counter'][rew_index] = self.goalkeeper_actions['initiate_counter']
                    reward[rew_index] += components['initiate_counter'][rew_index]

        return reward, components

    def step(self, action):
        """
        Take a step using the underlying env, then adjust the reward using the defined scheme.
        """
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
