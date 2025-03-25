import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that promotes energy conservation through proficient usage 
    of Stop-Sprint and Stop-Moving actions to manage player stamina and 
    positional integrity during a football match."""

    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        # Parameters for rewards
        self.stamina_saving_bonus = 0.01  # Reward for not sprinting
        self.movement_efficiency_bonus = 0.005  # Reward for minimal unnecessary movement
        self.previous_sticky_actions = None  # To keep track of previous actions

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.previous_sticky_actions = None
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['sticky_actions_counter'] = self.sticky_actions_counter
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['sticky_actions_counter']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()

        if observation is None:
            return reward

        components = {"base_score_reward": reward.copy(),
                      "stamina_saving_reward": [0.0] * len(reward),
                      "movement_efficiency_reward": [0.0] * len(reward)}

        for rew_index in range(len(reward)):
            current_sticky_actions = observation[rew_index]['sticky_actions']

            # Check for sprinting activity
            if self.previous_sticky_actions is not None:
                if self.previous_sticky_actions[rew_index][8] == 1 and current_sticky_actions[8] == 0:
                    # Reward for stopping sprint
                    components["stamina_saving_reward"][rew_index] = self.stamina_saving_bonus
                    self.sticky_actions_counter[8] += 1
                
                # Calculate inactivity or minimal movement
                action_changes = np.sum(self.previous_sticky_actions[rew_index] != current_sticky_actions)
                if action_changes == 0 or (current_sticky_actions[0:8].sum() == 0 and current_sticky_actions[9] == 0):
                    # Bonus for sustain position without unnecessary movements
                    components["movement_efficiency_reward"][rew_index] += self.movement_efficiency_bonus

            reward[rew_index] += (components["stamina_saving_reward"][rew_index] +
                                  components["movement_efficiency_reward"][rew_index])

        self.previous_sticky_actions = observation[:]['sticky_actions']

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
                self.sticky_actions_counter[i] += action
                info[f"sticky_actions_{i}"] = action
        return observation, reward, done, info
