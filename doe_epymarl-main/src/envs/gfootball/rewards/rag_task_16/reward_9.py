import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a reward for successful high passes, emphasizing accuracy, power, and scenario use."""
    
    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        # High pass action index for sticky_actions (assumed index for demonstration)
        self.high_pass_action_index = 10  # This index should be updated based on the environment definition
        self.previous_ball_height = 0.0
        self.pass_accuracy_threshold = 0.1  # Distance threshold to consider a pass 'accurate'
        self.min_pass_height = 0.15  # Minimum height considered a 'high pass'
        self.pass_reward_multiplier = 5.0

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.previous_ball_height = 0.0
        return self.env.reset()

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(), "high_pass_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        for rew_index in range(len(reward)):
            o = observation[rew_index]

            current_ball_height = o['ball'][2]
            ball_ownership = o['ball_owned_team'] == 1 and o['ball_owned_player'] == o['active']

            # Reward high passes: Check if jump in ball height occurs with a high pass action
            if (current_ball_height >= self.min_pass_height and
                current_ball_height > self.previous_ball_height and
                self.sticky_actions_counter[self.high_pass_action_index] == 1 and
                ball_ownership):
                
                # Check pass accuracy by measuring proximity to closest teammate
                teammate_positions = o['right_team'] if o['ball_owned_team'] == 1 else o['left_team']
                distances = np.linalg.norm(teammate_positions - o['ball'][:2], axis=1)
                min_distance = np.min(distances)

                if min_distance < self.pass_accuracy_threshold:
                    # Increase reward for accurate high passes
                    additional_reward = self.pass_reward_multiplier * (self.min_pass_height / current_ball_height)
                    components['high_pass_reward'][rew_index] = additional_reward
                    reward[rew_index] += additional_reward

            # Update for next check
            self.previous_ball_height = current_ball_height

        return reward, components

    def get_state(self, to_pickle):
        state = super().get_state(to_pickle)
        state['previous_ball_height'] = self.previous_ball_height
        return state

    def set_state(self, state):
        from_pickle = super().set_state(state)
        self.previous_ball_height = from_pickle['previous_ball_height']
        return from_pickle

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
