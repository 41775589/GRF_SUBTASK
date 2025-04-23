import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that provides a dense reward for actions leading to close-range
    attacks and successful dribbles against the goalkeeper.
    """

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)  # Storing sticky actions
        self._goal_approach_reward = 0.05
        self._successful_dribble_reward = 0.2
        self._shooting_reward = 0.3

    def reset(self):
        """
        Reset sticky action counters and other necessary components
        """
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        """
        Serialize customized state data if necessary
        """
        from_pickle = self.env.get_state(to_pickle)
        from_pickle['CheckpointRewardWrapper'] = None  # If any specific state needs saving
        return from_pickle

    def set_state(self, state):
        """
        Deserialize and set state data to environment
        """
        from_pickle = self.env.set_state(state)
        # Set any specific state handling if customization is restored
        return from_pickle

    def reward(self, reward):
        """
        Augment the base reward function by adding rewards for effective close-range play,
        dribbling and shooting.
        """
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "goal_approach_reward": [0.0] * len(reward),
                      "successful_dribble_reward": [0.0] * len(reward),
                      "shooting_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            if 'ball_owned_team' in o and 'ball_owned_player' in o and o['ball_owned_team'] == 0:
                
                # Check if the ball is near the opponent's goal or the agent is dribbling
                distance_to_goal = np.abs(o['ball'][0] - 1)
                if distance_to_goal < 0.2:
                    components["goal_approach_reward"][rew_index] = self._goal_approach_reward
                    reward[rew_index] += components["goal_approach_reward"][rew_index]

                last_action = self.sticky_actions_counter[9]  # dribble actions counter
                if last_action == 1:
                    components["successful_dribble_reward"][rew_index] = self._successful_dribble_reward
                    reward[rew_index] += components["successful_dribble_reward"][rew_index]

                last_action = self.sticky_actions_counter[7]  # shooting actions
                if last_action == 1:
                    components["shooting_reward"][rew_index] = self._shooting_reward
                    reward[rew_index] += components["shooting_reward"][rew_index]

        return reward, components

    def step(self, action):
        """
        Take an action using the wrapped environment, and adjust the reward
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
                # Update sticky actions counter
                if action > 0:
                    self.sticky_actions_counter[i] += 1
        return observation, reward, done, info
