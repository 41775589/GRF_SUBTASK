import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that encourages quick decision-making for counter-attacks."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        # No custom states in this example, this would persist custom data across deployments
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        # No custom states in this example, this would restore custom data
        return from_pickle

    def reward(self, reward):
        # Placeholders for reward components and the observation access
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy()}

        # Add new reward components initialized to zero
        components["quick_decision_bonus"] = np.zeros_like(reward)
        components["efficient_ball_handling_bonus"] = np.zeros_like(reward)

        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for i in range(len(reward)):
            o = observation[i]

            # Encourage quick decisions: reward added if the player performs a kicking action shortly after ball recovery
            if o['ball_owned_team'] == 0:  # Assuming team index 0 is the controlled team
                components["quick_decision_bonus"][i] = 0.02  # small bonus for having the ball, encouraging quick play

                # Check if player changed control of the ball to their team in this step
                if o['ball_owned_player'] == o['designated'] and o['ball_owned_player'] != -1:
                    components["quick_decision_bonus"][i] += 0.1  # bigger bonus for recovering the ball

            # Efficient ball handling: checks sticky actions that indicate effective ball control
            if o['sticky_actions'][9] == 1:  # Assuming index 9 is the dribble action
                components["efficient_ball_handling_bonus"][i] = 0.01  # small bonus for dribbling

            # Combine the rewards with the components
            reward[i] += components["quick_decision_bonus"][i] + components["efficient_ball_handling_bonus"][i]

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        return observation, reward, done, info
