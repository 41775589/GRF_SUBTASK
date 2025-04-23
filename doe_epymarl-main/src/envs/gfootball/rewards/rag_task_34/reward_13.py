import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense reward for quick decision-making and precision
       in close-range attacks, focusing on dribbling and shooting against goalkeepers."""
    
    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self._goal_attempts = 0
        self._successful_dribbles = 0
        
    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self._goal_attempts = 0
        self._successful_dribbles = 0
        return self.env.reset()
    
    def get_state(self, to_pickle):
        to_pickle['goal_attempts'] = self._goal_attempts
        to_pickle['successful_dribbles'] = self._successful_dribbles
        return self.env.get_state(to_pickle)
    
    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self._goal_attempts = from_pickle['goal_attempts']
        self._successful_dribbles = from_pickle['successful_dribbles']
        return from_pickle
    
    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "goal_attempt_reward": [0.0] * len(reward),
            "dribble_success_reward": [0.0] * len(reward)
        }
        
        if observation is None:
            return reward, components
        
        for rew_index in range(len(reward)):
            o = observation[rew_index]
            if o['game_mode'] in {0, 1}:  # Normal play or Kick-off
                if o['ball_owned_team'] == 1 and o['ball_owned_player'] == o['active']:  # Ball owned by the agent
                    goal_distance = abs(o['ball'][0] - 1)  # Distance to the opponent's goal along x
                    if goal_distance < 0.2:  # Close to the goal
                        self._goal_attempts += 1
                        components["goal_attempt_reward"][rew_index] = 1.0  # Reward for shot attempts near the goal

                    dribbling = any(o['sticky_actions'][8:10])  # Checking if dribbling actions are active
                    if dribbling:
                        self._successful_dribbles += 1
                        components["dribble_success_reward"][rew_index] = 0.5  # Reward for successful dribbling

        for i in range(len(reward)):
            reward[i] += (components["goal_attempt_reward"][i] + components["dribble_success_reward"][i])
        
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
