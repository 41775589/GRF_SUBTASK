import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense reward for promoting offensive plays."""
    
    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        """Reset the environment and the counter for sticky actions."""
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        """Get the state of the environment including this wrapper's specifics."""
        to_pickle['CheckpointRewardWrapper'] = {}
        return self.env.get_state(to_pickle)

    def set_state(self, from_pickle):
        """Set the state of the environment including this wrapper's specifics."""
        state = self.env.set_state(from_pickle)
        return state

    def reward(self, reward):
        """Calculate dense rewards for promoting offensive plays and pressure handling."""
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(), "offensive_play_reward": [0.0] * len(reward)}
        
        if observation is None:
            return reward, components
        assert len(reward) == len(observation)
        
        for i in range(len(reward)):
            o = observation[i]
            
            # Check if the ball is close to the opponent's goal area to promote finishing skills
            if o['ball'][0] > 0.5:  # Assuming a field range -1 to 1 for X coordinate
                # Further increment for closer to goal and controlled by active player
                if o['ball_owned_team'] == 1 and o['ball_owned_player'] == o['active']:
                    components['offensive_play_reward'][i] += 0.4
                else:
                    components['offensive_play_reward'][i] += 0.2

            # Apply the rewards
            reward[i] += components['offensive_play_reward'][i]
        
        return reward, components

    def step(self, action):
        """Execute a step in the environment, augmenting with reward components."""
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
