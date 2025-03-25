import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that adds a complex checkpoint reward to encourage specific behaviors such as high passes, long passes,
    dribbling under pressure, sprinting, and abruptly stopping to adapt to game transitions.
    """

    def __init__(self, env):
        super().__init__(env)
        # Initialize counters for different actions - the actions you've indicated interest in.
        self.sprint_counter = np.zeros(10, dtype=int)
        self.dribble_under_pressure_counter = np.zeros(10, dtype=int)
        self.high_pass_counter = np.zeros(10, dtype=int)
        self.long_pass_counter = np.zeros(10, dtype=int)
        # Constants for additional rewards per action.
        self.reward_for_high_pass = 0.2
        self.reward_for_long_pass = 0.3
        self.reward_for_dribble_under_pressure = 0.4
        self.reward_for_sprint_transition = 0.1

    def reset(self):
        # Reset action counters on environment reset.
        self.sprint_counter.fill(0)
        self.dribble_under_pressure_counter.fill(0)
        self.high_pass_counter.fill(0)
        self.long_pass_counter.fill(0)
        return self.env.reset()

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        if observation is None:
            return reward, {}

        components = {'base_score_reward': reward.copy(),
                      'high_pass_reward': [0.0] * len(reward),
                      'long_pass_reward': [0.0] * len(reward),
                      'dribble_reward': [0.0] * len(reward),
                      'sprint_transition_reward': [0.0] * len(reward)}

        # Iterate over each agent's observation.
        for i in range(len(observation)):
            o = observation[i]
            if o['ball_owned_team'] == 1:  # Check if right team controls the ball.
                # Check for specific sticky actions.
                if o['sticky_actions'][7] and self.high_pass_counter[i] == 0: # High pass action.
                    reward[i] += self.reward_for_high_pass
                    components['high_pass_reward'][i] = self.reward_for_high_pass
                    self.high_pass_counter[i] = 1

                if o['sticky_actions'][6] and self.long_pass_counter[i] == 0:  # Long pass action.
                    reward[i] += self.reward_for_long_pass
                    components['long_pass_reward'][i] = self.reward_for_long_pass
                    self.long_pass_counter[i] = 1
            
            # Check for dribbling under pressure.
            if ('ball_owned_team' in o) and (o['ball_owned_team'] == o['active']) and self.dribble_under_pressure_counter[i] == 0:
                reward[i] += self.reward_for_dribble_under_pressure
                components['dribble_reward'][i] = self.reward_for_dribble_under_pressure
                self.dribble_under_pressure_counter[i] = 1

            # Transition between sprint and stop sprint.
            if o['sticky_actions'][8] and not self.sprint_counter[i]:  # Sprint action.
                self.sprint_counter[i] = 1
            elif not o['sticky_actions'][8] and self.sprint_counter[i]:  # Stop Sprint action.
                reward[i] += self.reward_for_sprint_transition
                components['sprint_transition_reward'][i] = self.reward_for_sprint_transition
                self.sprint_counter[i] = 0

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info['final_reward'] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        return observation, reward, done, info
