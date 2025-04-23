import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that adds a dense reward component emphasizing ball control and passing 
    accuracy under pressure in tight game situations.
    """
    def __init__(self, env):
        super().__init__(env)
        self.pass_control_bonus = 0.05
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {'sticky_actions_counter': self.sticky_actions_counter}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['CheckpointRewardWrapper']['sticky_actions_counter']
        return from_pickle

    def reward(self, reward):
        # Get the current state of the environment
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "pass_control_bonus": [0.0, 0.0]}

        if observation is None:
            return reward, components

        for index, obs in enumerate(observation):
            # Check if the player is in control and under pressure
            if obs['ball_owned_team'] == obs['active'] and np.sum(obs['left_team_active']) + np.sum(obs['right_team_active']) > 8:
                # Extract the appropriate sticky_actions indices for passes:
                # 2 = action_top (treated as a high pass, hypothetically)
                # 3 = action_top_right (treated as a long pass, hypothetically)
                # 5 = action_bottom_right (treated as a short pass, hypothetically)
                pass_actions = (obs['sticky_actions'][2], obs['sticky_actions'][3], obs['sticky_actions'][5])
                if any(pass_actions):
                    # If any pass action is active, provide a small bonus
                    components['pass_control_bonus'][index] = self.pass_control_bonus
                    reward[index] += self.pass_control_bonus

        return reward, components

    def step(self, action):
        # Standard step process using the parent environment
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        
        # Adding reward components into info for debugging purposes
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                self.sticky_actions_counter[i] = action
        
        return observation, reward, done, info
