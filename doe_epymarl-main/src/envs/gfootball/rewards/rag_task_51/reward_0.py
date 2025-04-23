import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that adds a specialized reward for goalkeeper training, focusing on
    shot stopping, reflexes, and initiating counter-attacks.
    """
    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "goalkeeper_reflex_reward": [0.0] * len(reward),
                      "counter_attack_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]

            # Base reward includes initial reward from gameplay
            components["base_score_reward"][rew_index] = reward[rew_index]

            # Define roles
            goalkeeper_role = 0

            # Check whether the active player is the goalkeeper
            if o['active'] == o['designated'] and o['left_team_roles'][o['active']] == goalkeeper_role:
                # Reward quick reflexes and shot stopping
                if o['game_mode'] in [3, 6]:  # FreeKick or Penalty
                    # Encourage the goalie to be ready
                    components["goalkeeper_reflex_reward"][rew_index] = 0.5
                    reward[rew_index] += components["goalkeeper_reflex_reward"][rew_index]

                # Reward counter-attack capability
                if o['game_mode'] == 0:  # Normal play
                    if np.linalg.norm(o['ball']) < 0.1 and o['ball_owned_team'] == 0:  # ball is close and owned by the left team (goalie's team)
                        components["counter_attack_reward"][rew_index] = 1.0
                        reward[rew_index] += components["counter_attack_reward"][rew_index]

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)

        # Add all component values to info for diagnostic purposes
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)

        # Also update sticky_actions counter for info
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                info[f"sticky_actions_{i}"] = action
        
        return observation, reward, done, info
