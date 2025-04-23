import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a specific reward for precise defensive transitions in action states to counter various offensive plays."""

    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['sticky_actions_counter'] = self.sticky_actions_counter.tolist()
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = np.array(from_pickle['sticky_actions_counter'])
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "transition_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            active_player_sticky_actions = o['sticky_actions']
            
            # Calculate transition rewards based on moving and stopping frequently to adjust defensive positions
            # checking changes in the 'action_sprint' and 'action_idle' sticky actions
            sprint_index = 8
            idle_index = 9
            current_sprint_idle = active_player_sticky_actions[sprint_index] + active_player_sticky_actions[idle_index]
            previous_sprint_idle = self.sticky_actions_counter[sprint_index] + self.sticky_actions_counter[idle_index]

            if current_sprint_idle > previous_sprint_idle:  # if transition happened
                components["transition_reward"][rew_index] += 0.05

            reward[rew_index] += components["transition_reward"][rew_index]

            # Update sticky_actions_counter for next reward processing
            self.sticky_actions_counter = active_player_sticky_actions

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        
        # Store actions states for tracking transitions
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                self.sticky_actions_counter[i] = action
        
        return observation, reward, done, info
