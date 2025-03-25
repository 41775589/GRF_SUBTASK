import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A Gym wrapper that adds rewards for shooting accuracy, effective dribbling, and mastering passes.
    It promotes offensive strategies.
    """
    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        # Initialize counters for the actions we are interested in
        self.ball_control_counter = 0
        self.pass_accuracy_counter = 0
        self.shoot_accuracy_counter = 0
        self.sticky_actions_counter = np.zeros(10, dtype=int)  # For dribbling tracking

        # Define reward increments for specific actions
        self.control_reward = 0.05
        self.pass_reward = 0.1
        self.shoot_reward = 0.2

    def reset(self):
        # Reset the action counters on a new episode
        self.ball_control_counter = 0
        self.pass_accuracy_counter = 0
        self.shoot_accuracy_counter = 0
        self.sticky_actions_counter.fill(0)
        return self.env.reset()

    def get_state(self, to_pickle):
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        return from_pickle

    def reward(self, reward):
        """
        Augments rewards based on control of the ball, passing, and shooting.
        """
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "control_reward": [0.0] * len(reward),
            "pass_reward": [0.0] * len(reward),
            "shoot_reward": [0.0] * len(reward)
        }

        for i, rew in enumerate(reward):
            o = observation[i]
            # Check if the active player has the ball
            if o['ball_owned_team'] == 0 and o['ball_owned_player'] == o['active']:
                # Player is controlling the ball
                self.ball_control_counter += 1
                components['control_reward'][i] = self.ball_control_counter * self.control_reward

                # Check for shooting condition
                if 'action' in o and o['action'] == 'action_shot':
                    self.shoot_accuracy_counter += 1
                    components['shoot_reward'][i] = self.shoot_accuracy_counter * self.shoot_reward

                # Check for passing condition
                if 'action' in o and o['action'] in ['action_long_pass', 'action_high_pass']:
                    self.pass_accuracy_counter += 1
                    components['pass_reward'][i] = self.pass_accuracy_counter * self.pass_reward

            # Aggregate all components into the total reward
            reward[i] += (components['control_reward'][i] +
                          components['pass_reward'][i] +
                          components['shoot_reward'][i])
        
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
