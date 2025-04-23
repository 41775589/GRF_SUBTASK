import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a defensive positional reward to enhance defensive unit's capability to handle direct attacks through improved confrontational defense and strategic positioning for counterattacks."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.reset()

    def reset(self):
        """
        Resets the environment, and also resets the sticky actions counter.
        """
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        """Saves the current state of the environment along with this wrapper's specific state."""
        to_pickle['sticky_actions_counter'] = self.sticky_actions_counter
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """Restores the state of the environment as well as this wrapper's specific state."""
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['sticky_actions_counter']
        return from_pickle

    def reward(self, reward):
        """
        Modifies the rewards given by the environment based on the agent's ability to effectively position themselves defensively.
        """
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": list(reward)}

        if observation is None:
            return reward, components

        for i, o in enumerate(observation):
            # Enhance reward based on defensive positioning and ball interception potential
            if o['ball_owned_team'] == 1 and o['designated'] == o['active']:
                # Ball is with opponent team
                distance_to_opponent = np.linalg.norm(o['left_team'] - o['right_team'][o['designated']])
                # Reward intercepting the ball or being close to it
                reward[i] += 0.1 * (1 - min(distance_to_opponent, 1))

            if o['game_mode'] == 4 and o['ball_owned_team'] == 0:
                # Our team just got the ball on a counterattack opportunity
                reward[i] += 0.5

            if o['game_mode'] in {2, 3, 5}:
                # Defensive mode on goal kick, free kick, or throw in:
                reward[i] += 0.3

        return list(reward), components

    def step(self, action):
        """
        Takes a step in the environment, applying the action, and then modifies the reward using the custom reward function.
        """
        observation, reward, done, info = self.env.step(action)
        modified_reward, components = self.reward(reward)
        info['final_reward'] = sum(modified_reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                info[f"sticky_actions_{i}"] = action
        return observation, modified_reward, done, info
