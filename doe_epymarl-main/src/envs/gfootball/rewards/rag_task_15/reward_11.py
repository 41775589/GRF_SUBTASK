import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense reward for mastering long passes in the football game."""

    def __init__(self, env):
        super().__init__(env)
        self.target_positions = [
            (0.7, 0), (0.9, 0.3), (0.9, -0.3), # Long forward and angled passes
            (0.0, 0.42), (0.0, -0.42)  # Cross field long passes
        ]
        self.pass_accuracy_reward = 0.2
        self.pass_completion_reward = 0.5
        self.reset_pass_counters()

    def reset_pass_counters(self):
        self.passes_completed = {pos: False for pos in self.target_positions}
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.reset_pass_counters()
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = self.passes_completed
        return self.env.get_state(to_pickle)
    
    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.passes_completed = from_pickle['CheckpointRewardWrapper']
        return from_pickle
    
    def reward(self, reward):
        base_reward = reward.copy()
        additional_reward = [0.0] * len(reward)
        observation = self.env.unwrapped.observation()
        
        if observation is None:
            return reward, {}

        for i, o in enumerate(observation):
            if o['ball_owned_team'] == 0:  # Assuming team 0 is the controlled team
                ball_pos = o['ball'][:2]  # Get the x, y position of the ball
                prev_ball_pos = ball_pos - o['ball_direction'][:2]
                for target_pos in self.target_positions:
                    dist_to_target = np.linalg.norm(np.array(target_pos) - np.array(ball_pos))
                    if dist_to_target < 0.1:
                        if not self.passes_completed[target_pos]:
                            additional_reward[i] += self.pass_accuracy_reward
                            self.passes_completed[target_pos] = True
                    if np.linalg.norm(np.array(target_pos) - np.array(prev_ball_pos)) > dist_to_target:
                        additional_reward[i] += self.pass_completion_reward
        
        reward = [r + ar for r, ar in zip(base_reward, additional_reward)]
        components = {
            'base_score_reward': base_reward,
            'additional_reward': additional_reward
        }
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
