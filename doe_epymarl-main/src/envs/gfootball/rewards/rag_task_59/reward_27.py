import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper for adding a reward based on goalkeeper coordination and efficient clearances 
    to specific outfield players during high-pressure scenarios."""
    
    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.clearance_success = 0.2  # Reward for properly clearing the ball under pressure

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = self.sticky_actions_counter.tolist()
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = np.array(from_pickle['CheckpointRewardWrapper'])
        return from_pickle

    def reward(self, reward):
        """ This method specifies the added rewards for goalkeeper acting under pressure and 
        making successful clearances to designated outfield players."""
        observation = self.env.unwrapped.observation()
        base_score_reward = reward.copy()
        clearance_reward = [0.0] * len(reward)
        
        if observation is None:
            return reward, {"base_score_reward": base_score_reward, "clearance_reward": clearance_reward}
        
        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]

            # Check game mode for goalkeeper pressure contexts (e.g., corners, free kicks)
            if o['game_mode'] in [3, 4]:  # Free kicks or corners
                # Check if the goalkeeper or defenders make a clearance
                if o['ball_owned_team'] == 0 and o['ball_owned_player'] in o['left_team_roles'][:2]:  # GK or primary defenders
                    clearance_reward[rew_index] = self.clearance_success
                    reward[rew_index] += self.clearance_success

        return reward, {"base_score_reward": base_score_reward, "clearance_reward": clearance_reward}

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
                self.sticky_actions_counter[i] += action
                info[f"sticky_actions_{i}"] = action
        return observation, reward, done, info
