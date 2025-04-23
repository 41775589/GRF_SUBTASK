import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A custom reward wrapper emphasizing finishing techniques near the goal with optimal shooting angles and timing."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.goalscored = 0
        self.shoot_attempt_zone = 0.2  # The critical zone near the opponent's goal

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.goalscored = 0
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['goalscored'] = self.goalscored
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.goalscored = from_pickle.get('goalscored', 0)
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "shooting_reward": [0.0] * len(reward)}
        
        if observation is None:
            return reward, components

        for i, o in enumerate(observation):
            if 'score' in o:
                if o['score'][1] > self.goalscored:  # Assuming the agent's team is on the right
                    components["shooting_reward"][i] += 3.0  # High reward for scoring
                    self.goalscored = o['score'][1]
            if o['ball_owned_team'] == 1:  # ball owned by right team (agent's team)
                # Calculate ball proximity to opponent's goal
                ball_x, ball_y = o['ball'][0], o['ball'][1]
                if ball_x > (1 - self.shoot_attempt_zone):
                    # Ball is within the critical shooting zone
                    components["shooting_reward"][i] += 1.0  # Reward positioning for potential shooting

        for i in range(len(reward)):
            reward[i] += components["shooting_reward"][i]
        
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
