import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a specialized reward for practicing shots with varying shooting pressure."""

    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        # Keep track of the scoring attempts and shooting power applied
        self.previous_attempts = 0
        self.shooting_power = 0
        self.shot_reward = 1.0
        self.accuracy_bonus = 0.5

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.previous_attempts = 0
        self.shooting_power = 0
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {
            'previous_attempts': self.previous_attempts,
            'shooting_power': self.shooting_power
        }
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        state_info = from_pickle['CheckpointRewardWrapper']
        self.previous_attempts = state_info['previous_attempts']
        self.shooting_power = state_info['shooting_power']
        return from_pickle

    def reward(self, reward):
        # Get the current observation to adjust the reward based on shooting parameters
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "shooting_reward": [0.0] * len(reward)
        }

        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]

            # Rewarding shooting action with power and facing towards the goal
            if o['game_mode'] == 0:  # Checking in normal game mode
                if o['sticky_actions'][9] == 1:  # if 'shot' action is taken
                    self.previous_attempts += 1
                    distance_to_goal = np.linalg.norm(np.array([1, 0]) - o['ball'])
                    facing_goal = np.abs(o['ball_direction'][0]) > 0.5  # Rough metric for facing towards goal

                    shot_power = o['ball_direction'][2]  # z component reflects the shooting power somewhat
                    if facing_goal:
                        self.shooting_power += shot_power
                        components["shooting_reward"][rew_index] = self.shot_reward + self.accuracy_bonus * shot_power

            reward[rew_index] += components["shooting_reward"][rew_index]

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        # Adding the final reward and components to the info dictionary
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
