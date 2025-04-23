import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that adds a dense reward based on close-range attacks and quick decision-making against goalkeepers.
    """
    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)
        # Hyperparameters to adjust based on training needs
        self.goal_precision_reward = 0.5
        self.dribble_effectiveness_reward = 0.3
        self.decision_speed_reward = 0.2
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = self.sticky_actions_counter
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['CheckpointRewardWrapper']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "goal_precision": [0.0] * len(reward),
            "dribble_effectiveness": [0.0] * len(reward),
            "quick_decision": [0.0] * len(reward)
        }

        if observation is None:
            return reward, components

        for rew_index in range(len(reward)):
            obs = observation[rew_index]
            
            # Goal precision reward
            if obs['distance_to_goal'] < 0.1:  # Close range to the goal
                components["goal_precision"][rew_index] = self.goal_precision_reward

            # Dribble effectiveness
            dribbling = obs.get('sticky_actions', [0] * 10)[9]  # Assuming index 9 is dribble action
            if dribbling:
                components["dribble_effectiveness"][rew_index] = self.dribble_effectiveness_reward

            # Decision speed reward - Encourage quick actions
            current_steps = obs['steps_left']
            if current_steps > 0:
                time_factor = 1 - (current_steps / float(self._initial_steps))
                components["quick_decision"][rew_index] = self.decision_speed_reward * time_factor

            # Calculate the total modified reward
            reward[rew_index] += (components["goal_precision"][rew_index] +
                                  components["dribble_effectiveness"][rew_index] +
                                  components["quick_decision"][rew_index])

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
