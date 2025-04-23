import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that introduces reward adjustments focusing on enhanced defending
    strategies, encapsulating tackling proficiency, efficient movement control,
    and effective passing under pressure.
    """

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.tackle_coefficient = 0.5
        self.movement_control_coefficient = 0.3
        self.pressure_pass_coefficient = 0.6

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = True  # Additional state information can be added here
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        # You might want to load additional state information here when set state
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "tackle_reward": [0.0] * len(reward),
                      "movement_reward": [0.0] * len(reward),
                      "pressure_pass_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components
        
        assert len(reward) == len(observation)
        
        for rew_index in range(len(reward)):
            o = observation[rew_index]

            # Tackle-based reward (assumed tackling or interception when player rapidly approaches an opponent with the ball)
            if o['ball_owned_team'] == 1 and np.linalg.norm(o['ball'] - o['left_team'][o['designated']]) < 0.05:
                components["tackle_reward"][rew_index] = self.tackle_coefficient

            # Movement control (rewarded for less movement when near ball without contacting it)
            if np.linalg.norm(o['ball'] - o['left_team'][o['designated']]) < 0.1:
                components["movement_reward"][rew_index] = self.movement_control_coefficient * (1 - np.sum(np.abs(o['left_team_direction'][o['designated']])))
            
            # Pressure passing reward (when passing under defensive pressure)
            if o['ball_owned_team'] == 0 and o['left_team_tired_factor'][o['designated']] > 0.5:
                components["pressure_pass_reward"][rew_index] = self.pressure_pass_coefficient

            # Accumulate additional rewards to the main reward component
            reward[rew_index] += (components["tackle_reward"][rew_index] +
                                  components["movement_reward"][rew_index] +
                                  components["pressure_pass_reward"][rew_index])

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
