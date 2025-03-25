import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that rewards mastering High Pass and wide midfield responsibilities."""
    
    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        
        # Number of lateral high-pass actions successfully completed
        self.lateral_high_pass_counter = [0, 0]

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.lateral_high_pass_counter = [0, 0]
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['lateral_high_pass_counter'] = self.lateral_high_pass_counter
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.lateral_high_pass_counter = from_pickle['lateral_high_pass_counter']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(), "positioning_reward": [0.0] * len(reward)}
        
        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for idx, obs in enumerate(observation):
            is_left_side = obs['right_team'][obs['active']][0] < 0
            player_with_ball = obs['ball_owned_player'] == obs['active']
            is_high_pass = self.sticky_actions_counter[5] == 1  # assuming index 5 is the high pass
            
            # Encourage lateral passing and positioning on the wide areas of the field 
            if player_with_ball and is_high_pass and abs(obs['right_team'][obs['active']][1]) > 0.2:
                if is_left_side:
                    # Player is on the left and passes to the right wide area
                    if obs['ball'][0] - obs['right_team'][obs['active']][0] > 0.5:
                        components["positioning_reward"][idx] = 0.3
                        self.lateral_high_pass_counter[idx] += 1
                else:
                    # Player is on the right and passes to the left wide area
                    if obs['ball'][0] - obs['right_team'][obs['active']][0] < -0.5:
                        components["positioning_reward"][idx] = 0.3
                        self.lateral_high_pass_counter[idx] += 1
            
            # Combine the rewards
            reward[idx] = reward[idx] + components["positioning_reward"][idx]

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
            for i, act in enumerate(agent_obs['sticky_actions']):
                self.sticky_actions_counter[i] += act
                info[f"sticky_actions_{i}"] = act
        return observation, reward, done, info
