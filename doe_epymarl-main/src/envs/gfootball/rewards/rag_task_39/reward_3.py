import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that adds a dense reward for clearing the ball effectively from the defensive zone under pressure.
    This rewards defensive plays where the player effectively clears the ball when close to their own goal and under pressure from opponents.
    """
    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        # Relative distance under which the player is considered close to their own goal.
        self.close_to_goal_threshold = -0.5
        # Relative distance under which the ball is considered cleared.
        self.clearance_threshold = 0.2
        self.clearance_reward = 0.5

    def reset(self):
        """
        Reset the reward wrapper state for a new episode.
        """
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        """
        Get the state of the reward wrapper, allows saving and resuming.
        """
        to_pickle['CheckpointRewardWrapper'] = {}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """
        Set the state of the reward wrapper, allows loading saved states.
        """
        from_pickle = self.env.set_state(state)
        return from_pickle

    def reward(self, reward):
        """
        Augment the environment reward based on the agent's clearance effectiveness.
        """
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "clearance_reward": [0.0] * len(reward)}

        for rew_index in range(len(reward)):
            o = observation[rew_index]

            ball_pos_x = o['ball'][0]
            if o['ball_owned_team'] == 0 and ball_pos_x < self.close_to_goal_threshold:
                player_pos_x = o['left_team'][o['active']][0]
                # Check if clearance is successful.
                if ball_pos_x >= player_pos_x + self.clearance_threshold:
                    components["clearance_reward"][rew_index] = self.clearance_reward
                    reward[rew_index] += components["clearance_reward"][rew_index]

        return reward, components

    def step(self, action):
        """
        Perform an environment step with the given action, apply reward transformation, and provide detailed reward components.
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
                self.sticky_actions_counter[action] += 1
        for i in range(len(self.sticky_actions_counter)):
            info[f"sticky_actions_{i}"] = self.sticky_actions_counter[i]
        return observation, reward, done, info
