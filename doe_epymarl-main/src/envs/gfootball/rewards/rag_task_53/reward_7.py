import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that encourages maintaining ball control, strategic plays, and ball distribution."""
    
    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.ball_control_reward_multiplier = 0.05
        self.game_mode_bonus = 0.1
        
    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['checkpoint_reward_env_state'] = {}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        # Restore any internal state if necessary
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "ball_control": [0.0] * len(reward),
            "game_mode_bonus": [0.0] * len(reward)
        }

        if observation is None:
            return reward, components
        
        for rew_index in range(len(reward)):
            o = observation[rew_index]

            # Reward for ball possession
            if o['ball_owned_team'] == 0 and o['active'] == o['ball_owned_player']:
                components["ball_control"][rew_index] += self.ball_control_reward_multiplier * np.exp(-0.1*o['ball'][2])

            # Bonus for advantageous game modes (like when in Free Kick or Corner)
            if o['game_mode'] in [3, 4]:
                components["game_mode_bonus"][rew_index] = self.game_mode_bonus
            
            # Combine components to total reward
            reward[rew_index] += sum(components[k][rew_index] for k in components)
        
        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        
        self.sticky_actions_counter.fill(0)
        obs = self.env.unwrapped.observation()
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                info[f"sticky_actions_{i}"] = action
        
        return observation, reward, done, info
