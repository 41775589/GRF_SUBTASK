import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that adds a dense reward for a specific task in defense,
    focusing on the role of a 'sweeper'.
    """

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['sticky_actions_counter'] = self.sticky_actions_counter
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['sticky_actions_counter']
        return from_pickle

    def reward(self, reward):
        # Extend the base reward by additional metrics
        components = {"base_score_reward": reward.copy(),
                      "tackle_reward": [0.0] * len(reward),
                      "clearance_reward": [0.0] * len(reward)}

        observation = self.env.unwrapped.observation()
        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]

            # Reward based on successfully clearing the ball from the defensive zone
            if o['game_mode'] in {2, 3, 4}:  # Game modes related to defense situations
                if 'ball_owned_team' in o and o['ball_owned_team'] == 0 and 'ball' in o:
                    if o['ball'][0] < -0.7: # The ball is in the defensive third
                        components["clearance_reward"][rew_index] = 0.5  # Positive reward for clearance

            # Reward based on executing effective tackles as the last man
            if o['game_mode'] in {0}:  # Normal game mode
                if 'ball_owned_team' in o and o['ball_owned_team'] == 1:  # Ball owned by opposing team
                    if 'right_team_active' in o and o['right_team_active'][o['active']]:
                        # Tackle leading to change in ball possession
                        if 'left_team' in o and o['left_team'][o['active']][0] < -0.9:  # Close to own goal
                            components["tackle_reward"][rew_index] = 1.0  # High reward for last man tackle

            # Combine all components
            reward[rew_index] += (components["tackle_reward"][rew_index] +
                                  components["clearance_reward"][rew_index])

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
