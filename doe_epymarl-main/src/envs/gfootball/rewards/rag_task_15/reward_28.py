import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that focuses on rewarding accurate long passes in different game conditions.
    The task includes understanding and mastering the precision of long passes under varying match conditions.
    """
    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.pass_length_threshold = 0.5  # Define a threshold for what is considered a 'long pass'
        self.pass_accuracy_reward = 0.5  # Reward for making a precise pass
        self.observed_passes = {}  # Store observations of passes to check changes

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.observed_passes.clear()
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = self.observed_passes
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.observed_passes = from_pickle['CheckpointRewardWrapper']
        self.reset()
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "long_pass_reward": [0.0] * len(reward)}
        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            components["base_score_reward"][rew_index] = reward[rew_index]

            if not self.observed_passes.get(rew_index):
                self.observed_passes[rew_index] = {'previous_ball_position': None}
            
            current_ball_position = o['ball'][:2]  # Taking only x, y for simplicity
            previous_ball_position = self.observed_passes[rew_index]['previous_ball_position']
            
            if previous_ball_position and (o['ball_owned_team'] == 0 or o['ball_owned_team'] == 1):
                distance = np.linalg.norm(np.array(previous_ball_position) - np.array(current_ball_position))
                # Check if the ball was passed long distance and owned by the same team
                if distance > self.pass_length_threshold and o['ball_owned_player'] != -1:
                    components['long_pass_reward'][rew_index] = self.pass_accuracy_reward
                    reward[rew_index] += components['long_pass_reward'][rew_index]
                    
            # Update the ball position for next step calculation
            self.observed_passes[rew_index]['previous_ball_position'] = current_ball_position

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)

        # Update reward info for monitoring
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        
        # Update sticky actions count
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                self.sticky_actions_counter[i] += action
                info[f"sticky_actions_{i}"] = action

        return observation, reward, done, info
