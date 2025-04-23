import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that adds a customized reward for developing offensive strategies.
    Focuses primarily on mastering accurate shooting, effective dribbling, and executing different pass types.
    """

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        """
        Resets the environment and resets the sticky actions counter for the new episode.
        """
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        """
        Saves the sticky actions counter state.
        """
        to_pickle['CheckpointRewardWrapper_sticky_actions_counter'] = self.sticky_actions_counter
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """
        Restores the sticky actions counter state.
        """
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle.get('CheckpointRewardWrapper_sticky_actions_counter', np.zeros(10, dtype=int))
        return from_pickle

    def reward(self, reward):
        """
        Computes additional reward components based on offensive strategic play.
        """
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "pass_reward": [0.0] * len(reward),
                      "shoot_reward": [0.0] * len(reward),
                      "dribble_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        for rew_index in range(len(reward)):
            o = observation[rew_index]

            # Reward for successful shots at goal.
            if o['game_mode'] == 6 and o['ball_owned_team'] == 0:
                components["shoot_reward"][rew_index] = 0.3

            # Reward for effective dribbling (ball possession with sprint).
            if o['ball_owned_team'] == 0 and o['ball_owned_player'] == o['active'] and o['sticky_actions'][9]:
                components["dribble_reward"][rew_index] = 0.1
            
            # Reward for using long and high passes effectively.
            # Example condition: passing distance more than a threshold and successfully reaches a teammate.
            # Consider whether the ball direction and the next observation state confirm a successful pass.
            
            # Bonus for successful high/long passes.
            if o['ball_owned_team'] == 0 and (o['game_mode'] == 3 or o['game_mode'] == 4):
                components["pass_reward"][rew_index] = 0.2

            # Update the reward with the additional components
            reward[rew_index] += (components["shoot_reward"][rew_index] +
                                  components["dribble_reward"][rew_index] +
                                  components["pass_reward"][rew_index])

        return reward, components

    def step(self, action):
        """
        Takes a step in the environment and calculates the complete reward with additional components.
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
