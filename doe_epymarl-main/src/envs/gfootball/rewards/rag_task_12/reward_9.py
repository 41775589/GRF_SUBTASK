import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a tailored reward function for midfield/defensive training."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.passes_completed_counter = 0
        self.dribbles_made_counter = 0
        self.sprint_usage_counter = 0
        self.positioning_rewards = 0.05  # Reward increment for optimal positioning and actions.

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.passes_completed_counter = 0
        self.dribbles_made_counter = 0
        self.sprint_usage_counter = 0
        return self.env.reset()

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "passes_completed_reward": [0.0] * len(reward),
            "dribbling_reward": [0.0] * len(reward),
            "sprint_usage_reward": [0.0] * len(reward),
            "positioning_reward": [0.0] * len(reward)
        }

        if observation is None:
            return reward, components
        
        for i in range(len(reward)):
            o = observation[i]

            # Reward for successful high or long passes
            if o['sticky_actions'][8] == 1:  # Assuming index 8 is high pass in this context
                self.passes_completed_counter += 1
                components["passes_completed_reward"][i] = self.positioning_rewards
            
            if o['sticky_actions'][9] == 1:  # Assuming index 9 is long pass in this context
                self.passes_completed_counter += 1
                components["passes_completed_reward"][i] = self.positioning_rewards

            # Reward for maintaining dribbles under pressure
            if o['sticky_actions'][6]:  # Assuming index 6 is dribble
                self.dribbles_made_counter += 1
                components["dribbling_reward"][i] = self.positioning_rewards

            # Reward for smart sprint usage
            if o['sticky_actions'][7] and not o['sticky_actions'][0]:  # Sprint but not stop
                self.sprint_usage_counter += 1
                components["sprint_usage_reward"][i] = self.positioning_rewards
            elif o['sticky_actions'][0]:  # Stop sprint
                components["sprint_usage_reward"][i] = -self.positioning_rewards

            # Calculating the aggregate reward 
            reward[i] += (components["passes_completed_reward"][i] +
                          components["dribbling_reward"][i] +
                          components["sprint_usage_reward"][i] +
                          components["positioning_reward"][i])

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        self.sticky_actions_counter.fill(0)
        current_obs = self.env.unwrapped.observation()
        for agent_obs in current_obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                info[f"sticky_actions_{i}"] = action
        return observation, reward, done, info

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {
            "passes": self.passes_completed_counter,
            "dribbles": self.dribbles_made_counter,
            "sprints": self.sprint_usage_counter
        }
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        wrapper_state = from_pickle.get('CheckpointRewardWrapper', {})
        self.passes_completed_counter = wrapper_state.get("passes", 0)
        self.dribbles_made_counter = wrapper_state.get("dribbles", 0)
        self.sprint_usage_counter = wrapper_state.get("sprints", 0)
        return from_pickle
