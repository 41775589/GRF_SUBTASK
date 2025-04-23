import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that rewards agents for initiating counterattacks through long passes and quick transitions."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.long_pass_threshold = 0.3  # An arbitrary threshold for long passes
        self.quick_transition_time = 10  # Number of steps considered as 'quick' for making a transition

        # To keep track of defensive position and transition to attack
        self.last_ball_position_x = None
        self.transition_timer = np.inf  # Set to a large value initially

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.last_ball_position_x = None
        self.transition_timer = np.inf
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['last_ball_position_x'] = self.last_ball_position_x
        to_pickle['transition_timer'] = self.transition_timer
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.last_ball_position_x = from_pickle.get('last_ball_position_x', None)
        self.transition_timer = from_pickle.get('transition_timer', np.inf)
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "transition_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for idx, o in enumerate(observation):
            # Initial conditions to possibly start a counterattack
            if o['ball_owned_team'] == 0 and self.last_ball_position_x is not None:
                # Calculate the displacement of the ball along the x-axis
                ball_displacement_x = abs(o['ball'][0] - self.last_ball_position_x)
                
                # Detect a long pass (large displacement in x-direction)
                if ball_displacement_x > self.long_pass_threshold:
                    self.transition_timer = 0  # Reset transition timer upon detecting a long pass

            # Check transition from defense to attack
            if self.transition_timer <= self.quick_transition_time:
                components["transition_reward"][idx] += 5.0  # Reward for quick transition

            reward[idx] += components["transition_reward"][idx]

            # Sticky actions decrement
            for action_idx in range(10):
                if o['sticky_actions'][action_idx]:
                    self.sticky_actions_counter[action_idx] += 1

            # Update for next step
            self.last_ball_position_x = o['ball'][0]
            if self.transition_timer < np.inf:
                self.transition_timer += 1

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        return observation, reward, done, info
