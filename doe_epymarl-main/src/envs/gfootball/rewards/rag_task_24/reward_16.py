import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that encourages mid to long-range passing effectiveness and strategic use of high and long passes."""
    
    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        # Define the importance of pass range and accuracy
        self.pass_accuracy_reward = 0.05
        self.long_pass_reward = 0.1

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "pass_accuracy_reward": [0.0] * len(reward),
                      "long_pass_reward": [0.0] * len(reward)}
                      
        if observation is None:
            return reward, components
        
        assert len(reward) == len(observation)

        for i in range(len(reward)):
            obs = observation[i]
            if 'ball_owned_team' in obs and obs['ball_owned_team'] in [-1, 1]:
                # The ball is either not owned or owned by the opponent, skip further checks
                continue
            
            if 'ball_owned_player' in obs and obs['ball_owned_player'] == obs['active']:
                # Check if the controlled player owns the ball
                player_pos = obs['left_team'][obs['active']]
                ball_end_pos = obs['ball'][:2]  # Assuming the ball positon is updated due to the kick
                distance = np.linalg.norm(player_pos - ball_end_pos)
                
                # Reward longer passes with both distance and direction into account
                if distance > 0.3:  # somewhat arbitrary threshold for long pass
                    components['long_pass_reward'][i] = self.long_pass_reward
                    reward[i] += components['long_pass_reward'][i]
                
                # Analyze pass accuracy based on the final location of the ball
                # Assuming hypothetical 'target zones' which would be strategically beneficial
                target_zones = [np.array([1.0, 0.0]), np.array([1.0, 0.4]), np.array([1.0, -0.4])]
                accuracy = min([np.linalg.norm(ball_end_pos - target) for target in target_zones])
                
                if accuracy < 0.1:  # Closer to target is better
                    components['pass_accuracy_reward'][i] = self.pass_accuracy_reward / accuracy
                    reward[i] += components['pass_accuracy_reward'][i]
        
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
