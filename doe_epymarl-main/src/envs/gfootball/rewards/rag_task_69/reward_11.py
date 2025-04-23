import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense reward for offensive strategies."""

    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.passing_threshold = 0.5  # Threshold distances for considering good long/high passes
        self.passing_reward = 0.5     # Extra reward for effective passing
        self.dribbling_reward = 0.3   # Extra reward for effective dribbling

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper_sticky_actions_counter'] = self.sticky_actions_counter
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['CheckpointRewardWrapper_sticky_actions_counter']
        return from_pickle

    def reward(self, reward):
        """
        Enhances the base reward by adding bonuses for exhibiting skills related to offensive strategies:
        - Accurate long and high passes
        - Effective dribbling to evade opponents
        """
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "passing_reward": [0.0] * len(reward),
                      "dribbling_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            
            # Reward for making successful long/high passes
            if o['game_mode'] in [5, 6]:  # Assuming game modes for long and high passes respectively
                start_ball_pos = o['ball']  # During the start of action
                steps_after_action = 1      # This should ideally be calculated or observed after action
                end_ball_pos = self.env.unwrapped.observation()[rew_index]['ball']
                
                distance = np.linalg.norm(np.array(end_ball_pos[:2]) - np.array(start_ball_pos[:2]))
                if distance > self.passing_threshold:
                    components['passing_reward'][rew_index] = self.passing_reward
                    reward[rew_index] += components['passing_reward'][rew_index]
            
            # Reward for dribbling: effective if retains ball possession and evades near opponents
            if o['ball_owned_player'] == o['active'] and o['ball_owned_team'] == 0:
                if o['sticky_actions'][9] == 1:  # Dribble action is active
                    # Simple heuristic to check if any opponent is near
                    right_team_pos = o['right_team']
                    ball_pos = np.array(o['ball'][:2])
                    distances = np.linalg.norm(right_team_pos - ball_pos, axis=1)
                    
                    if np.any(distances < 0.3):  # Random threshold to decide if opponents are close enough
                        components['dribbling_reward'][rew_index] = self.dribbling_reward
                        reward[rew_index] += components['dribbling_reward'][rew_index]

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
                self.sticky_actions_counter[i] = action
        return observation, reward, done, info
