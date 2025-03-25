import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that modifies the behavior of the reward function to focus on the defensive tasks
    of a sweeper, such as clearing the ball from defensive zones, performing last-man tackles, and 
    covering positions efficiently.
    """
    def __init__(self, env):
        super().__init__(env)
        self.clearance_count = 0
        self.tackle_count = 0
        self.covering_bonus = 0
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.clearance_count = 0
        self.tackle_count = 0
        self.covering_bonus = 0
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        state_info = {'clearance_count': self.clearance_count,
                      'tackle_count': self.tackle_count,
                      'covering_bonus': self.covering_bonus}
        to_pickle['CheckpointRewardWrapper'] = state_info
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        state_info = from_pickle['CheckpointRewardWrapper']
        self.clearance_count = state_info['clearance_count']
        self.tackle_count = state_info['tackle_count']
        self.covering_bonus = state_info['covering_bonus']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "defensive_action_bonus": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            ball_position = o['ball']
            player_position = o['left_team'][o['active']] if o['ball_owned_team'] == 0 else o['right_team'][o['active']]
            
            # Reward for ball clearance from the defensive third
            if o['ball_owned_team'] == 1 and ball_position[0] < -0.5:
                self.clearance_count += 1
                components["defensive_action_bonus"][rew_index] += 0.2
            # Reward for successful tackles
            if o['game_mode'] == 3 and player_position[0] < -0.5:  # 3 is FreeKick mode, indicating a potential foul play
                self.tackle_count += 1
                components["defensive_action_bonus"][rew_index] += 0.1
            # Position covering efficiency bonus
            distance_to_ball = np.linalg.norm(np.array(ball_position[:2]) - np.array(player_position[:2]))
            if distance_to_ball < 0.1:
                self.covering_bonus += 0.01   # constant small bonus for staying close to defensive position
                components["defensive_action_bonus"][rew_index] += self.covering_bonus
            
            # Update the reward
            reward[rew_index] += components["defensive_action_bonus"][rew_index]
            
        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info['final_reward'] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                self.sticky_actions_counter[i] += action
        return observation, reward, done, info
