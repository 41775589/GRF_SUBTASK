import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a specialized reward for mastering defensive gameplay,
       focusing on positioning and interventions."""

    def __init__(self, env):
        super().__init__(env)
        # Track when the player intervenes positively in a high pressure defensive situation
        self.intercept_counter = np.zeros((4,))  # Assuming a maximum of 4 players
        self.previous_ball_owner = None
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.intercept_counter.fill(0)
        self.previous_ball_owner = None
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['intercept_counter'] = self.intercept_counter
        to_pickle['previous_ball_owner'] = self.previous_ball_owner
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.intercept_counter = from_pickle['intercept_counter']
        self.previous_ball_owner = from_pickle['previous_ball_owner']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "interception_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            current_ball_owner = (o['ball_owned_team'], o['ball_owned_player'])

            # Reward interception: if ball ownership changes from the opposing team to the agent team
            if self.previous_ball_owner and self.previous_ball_owner[0] != o['ball_owned_team']:
                if o['ball_owned_team'] == 0:  # Assuming the agent's team is 0
                    components["interception_reward"][rew_index] = 0.5
                    reward[rew_index] += components["interception_reward"][rew_index]
                    self.intercept_counter[rew_index] += 1

            # Store the current ball owner for use in the next cycle
            self.previous_ball_owner = current_ball_owner

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        info["interceptions"] = sum(self.intercept_counter)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        return observation, reward, done, info
