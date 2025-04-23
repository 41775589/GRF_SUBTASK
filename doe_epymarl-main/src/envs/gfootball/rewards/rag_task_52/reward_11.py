import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A gym RewardWrapper to modify the rewards for training specialized defending strategies 
    focusing on tackling proficiency, efficient movement control and pressured passing tactics.
    """
    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)  # Initialize sticky_actions counter
        
        # Coefficients for additional rewards
        self.tackle_reward_coefficient = 0.2  # Reward for successful tackles
        self.efficient_movement_coefficient = 0.1  # Reward for minimizing unnecessary movements
        self.pressured_pass_coefficient = 0.3  # Reward for successful passes under pressure

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)  # Reset sticky_actions counter at each reset
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = self.sticky_actions_counter
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['CheckpointRewardWrapper']
        return from_pickle

    def reward(self, reward):
        """
        Modifies the reward based on defending outcomes, tackling proficiency,
        movement efficiency and pressured passing performance.
        """
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(), 
            "tackle_reward": [0.0] * len(reward),
            "movement_efficiency_reward": [0.0] * len(reward),
            "pressured_pass_reward": [0.0] * len(reward)
        }
        
        if observation is None:
            return reward, components
        
        assert len(reward) == len(observation)

        for idx, o in enumerate(observation):
            # Check for tackles
            if 'ball_owned_team' in o and o['ball_owned_team'] == 0:  # Team 0 is defending
                components['tackle_reward'][idx] = self.tackle_reward_coefficient

            # Reward for stopping (considered when players are not moving significantly)
            if 'left_team_direction' in o:
                movement_norm = np.linalg.norm(o['left_team_direction'][o['active']], ord=2)
                if movement_norm < 0.01:  # Assuming low movement signifies good stopping control
                    components['movement_efficiency_reward'][idx] = self.efficient_movement_coefficient

            # Reward for successful pressured passes 
            # Checking game mode which normally changes on events like tackles or passes
            if o['game_mode'] in {2, 4, 6}:  # These game modes might involve pass under pressure scenarios
                components['pressured_pass_reward'][idx] = self.pressured_pass_coefficient

            # Aggregating rewards
            reward[idx] += (
                components['tackle_reward'][idx] + 
                components['movement_efficiency_reward'][idx] + 
                components['pressured_pass_reward'][idx]
            )

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        obs = self.env.unwrapped.observation()
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                self.sticky_actions_counter[i] = action  # Update the sticky_actions counter
        return observation, reward, done, info
