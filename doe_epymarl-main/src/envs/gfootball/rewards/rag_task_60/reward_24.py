import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a positional awareness and transition based reward function
       targeting defense improvements through started/stopped transitions."""
    
    def __init__(self, env):
        super().__init__(env)
        # Counter for the sticky actions (actions that have a lasting effect like sprint or dribble)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        # Reward increment parameters
        self.start_stop_reward = 0.1
        self.tired_penalty = -0.05

    def reset(self):
        """Reset sticky action counts and environment."""
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        """Return state including sticky action counts."""
        to_pickle['sticky_actions_counter'] = self.sticky_actions_counter
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """Set state including sticky action counts."""
        state = self.env.set_state(state)
        self.sticky_actions_counter = state.get('sticky_actions_counter', np.zeros(10, dtype=int))
        return state
        
    def reward(self, reward):
        """Calculate dense rewards based on positional awareness and transition states."""
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "transitional_reward": [0.0] * len(reward),
                      "tiredness_penalty": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            previous_actions = self.sticky_actions_counter.copy()

            # Update sticky action counters
            self.sticky_actions_counter = (o['sticky_actions'] > 0).astype(int)

            # Reward based on stopping and starting while in defensive posture
            if o['left_team_roles'][o['active']] in [0, 1, 2, 3, 4]:  # Defensive roles (GK, CB, LB, RB, DM)
                if previous_actions[8] != self.sticky_actions_counter[8]:  # Change in sprint status
                    components['transitional_reward'][rew_index] += self.start_stop_reward
                if previous_actions[9] != self.sticky_actions_counter[9]:  # Change in dribble status
                    components['transitional_reward'][rew_index] += self.start_stop_reward

            # Tiredness penalty for defensive players to avoid exhaustion
            components['tiredness_penalty'][rew_index] = o['left_team_tired_factor'][o['active']] * self.tired_penalty
            
            # Update the reward with components
            reward[rew_index] = reward[rew_index] + components['transitional_reward'][rew_index] + components['tiredness_penalty'][rew_index]

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info['final_reward'] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        return observation, reward, done, info
