import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that adjusts the reward system to focus on improving the defensive behaviors of agents against direct attacks
    through proficient defensive positioning, ball interception and setting up counterattacks, by giving positive reinforcement
    around crucial game situations.”
    """
    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        """Resets the environment and sticky actions counter."""
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        """Adds additional state information to the pickle related to this wrapper."""
        to_pickle['CheckpointRewardWrapper_sticky'] = self.sticky_actions_counter
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """Restores the state from the pickle including the sticky actions counter."""
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle.get('CheckpointRewardWrapper_sticky', np.zeros(10, dtype=int))
        return from_pickle

    def reward(self, reward):
        """
        Custom reward function focused on rewarding defensive positioning,
        ball interception likelihood, and setting up for effective counterattacks.
        """
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "defensive_positioning": [0.0] * len(reward),
                      "interception": [0.0] * len(reward),
                      "counter_setup": [0.0] * len(reward)}
        
        for rew_index in range(len(reward)):
            o = observation[rew_index]
            # Player is in a strategic position to intercept or block the ball
            if ('active' in o and o['ball_owned_team'] != o['active']): 
                components["defensive_positioning"][rew_index] = 0.2
            # Player intercepts the ball changing the flow of the game
            if o['ball_owned_team'] == 0 and o['ball_owned_player'] == o['active']:
                components["interception"][rew_index] = 0.5
            # Strategic play to setup a counterattack
            if any(action == 1 for action in o['sticky_actions'][6:8]): # Sprint or Dribble
                components["counter_setup"][rew_index] = 0.3
            # Cumulative reward calculation for defense
            reward[rew_index] += (components["defensive_positioning"][rew_index] +
                                  components["interception"][rew_index] +
                                  components["counter_setup"][rew_index])

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
