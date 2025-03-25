import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that introduces a reward for quick decision-making and efficient ball handling to
    initiate counter-attacks immediately after recovering possession.
    """

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.ball_possession_switched = False  # Flag to detect change in ball possession
        self.possession_start_step = 0
        self.last_reward_time = None  # To keep track of time since last ball recovery
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.recovery_speed_reward_coefficient = 0.2

    def reset(self):
        self.ball_possession_switched = False
        self.possession_start_step = 0
        self.last_reward_time = None
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CachedState'] = (self.ball_possession_switched, self.possession_start_step,
                                    self.last_reward_time)
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.ball_possession_switched, self.possession_start_step, self.last_reward_time = (
            from_pickle['CachedState'])
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        if observation is None or not any(reward):
            return reward, {}

        base_score_reward = reward.copy()
        
        components = {
            "base_score_reward": base_score_reward,
            "recovery_speed_reward": [0.0] * len(reward)
        }

        for i in range(len(reward)):
            current_observation = observation[i]

            # Check for ball possession switch to the current agent's team
            if current_observation['ball_owned_team'] == 0 and not self.ball_possession_switched:
                self.ball_possession_switched = True
                self.possession_start_step = current_observation['steps_left']
                self.last_reward_time = current_observation['steps_left']

            elif current_observation['ball_owned_team'] != 0:
                self.ball_possession_switched = False

            # Reward based on quick counter-attack initiation after gaining possession
            if self.ball_possession_switched:
                if current_observation['steps_left'] < self.last_reward_time:
                    time_since_possession = self.possession_start_step - current_observation['steps_left']
                    if time_since_possession <= 5:  # Reward if action taken within 5 steps
                        components["recovery_speed_reward"][i] = self.recovery_speed_reward_coefficient
                        reward[i] += components["recovery_speed_reward"][i]
                    self.last_reward_time = current_observation['steps_left']

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)

        # Update info dictionary with sticky actions counts
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                info[f"sticky_actions_{i}"] = action

        return observation, reward, done, info
