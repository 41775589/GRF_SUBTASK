import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds rewards for defensive actions."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        # Parameters related to defensive checkpoints
        self.interceptions_made = {}
        self.blocks_made = {}
        self.interception_reward = 0.5
        self.block_reward = 0.3

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.interceptions_made.clear()
        self.blocks_made.clear()
        return self.env.reset()

    def get_state(self, to_pickle):
        state = self.env.get_state(to_pickle)
        state['interceptions_made'] = self.interceptions_made
        state['blocks_made'] = self.blocks_made
        return state

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.interceptions_made = from_pickle['interceptions_made']
        self.blocks_made = from_pickle['blocks_made']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        
        components = {"base_score_reward": reward.copy(), 
                      "interception_reward": [0.0] * len(reward),
                      "block_reward": [0.0] * len(reward)}
        
        if observation is None:
            return reward, components

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            active_player = o['active']
            
            # Reward for interceptions
            if o['ball_owned_team'] == 1 and o['ball_owned_player'] == active_player:
                if rew_index not in self.interceptions_made:
                    self.interceptions_made[rew_index] = 1
                    reward[rew_index] += self.interception_reward
                    components['interception_reward'][rew_index] = self.interception_reward
            
            # Defense block -- approximate by not owning the ball but being very close to it
            ball_pos = o['ball'][:2] # x, y position
            player_pos = o['left_team'][active_player]

            # Calculate distance from the ball
            distance = np.linalg.norm(player_pos - ball_pos)
            if distance < 0.05 and o['ball_owned_team'] != 0:  # not owned by own team
                if rew_index not in self.blocks_made:
                    self.blocks_made[rew_index] = 1
                    reward[rew_index] += self.block_reward
                    components['block_reward'][rew_index] = self.block_reward
        
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
                info[f"sticky_actions_{i}"] = action
        return observation, reward, done, info
