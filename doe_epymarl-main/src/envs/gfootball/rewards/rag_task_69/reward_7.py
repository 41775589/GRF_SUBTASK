import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper modifies the rewards to focus on offensive strategies:
    accurate shooting, effective dribbling, and varied passing.
    Goals scored increase the reward, as well as successful pass completion 
    especially long and high passes, and dribbling near opponents without losing the ball.
    """
    
    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = self.sticky_actions_counter
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['CheckpointRewardWrapper']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "shooting_reward": [0.0] * len(reward),
            "dribbling_reward": [0.0] * len(reward),
            "passing_reward": [0.0] * len(reward)
        }

        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            # Component for shooting accuracy (goal scoring)
            if o['ball_owned_team'] == 0 and o['game_mode'] == 6: # game_mode 6 is Penalty
                components['shooting_reward'][rew_index] = 0.5  # High reward for scoring on a penalty shot

            # Component for dribbling - especially when evading nearby opponents
            if o['ball_owned_team'] == 0 and np.any(np.linalg.norm(o['left_team'] - o['right_team'][o['ball_owned_player']], axis=1) < 0.1):
                components["dribbling_reward"][rew_index] = 0.2  # Reward when dribbling close to opponents

            # Component for effective passing
            # Checking for long passes, using 'ball_direction' magnitude greater as indicator (simplified logic)
            if o['ball_owned_team'] == 0:
                if np.linalg.norm(o['ball_direction']) > 0.5:  # Assuming large ball direction indicates a long pass
                    components['passing_reward'][rew_index] = 0.3

            # Update overall reward for this index
            reward[rew_index] += components["shooting_reward"][rew_index] + components["dribbling_reward"][rew_index] + components["passing_reward"][rew_index]

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
                info[f"sticky_actions_{i}"] = action
        return observation, reward, done, info
