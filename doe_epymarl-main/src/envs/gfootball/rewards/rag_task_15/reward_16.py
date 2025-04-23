import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that adds a dense reward for technical aspects and precision of long passes.
    Long pass mastery involves understanding ball dynamics over various lengths and practicing under different conditions.
    The function reinforces practice with accurate passing over large distances across the football field.
    """

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        # Setting up threshold distances for long passes
        self.long_pass_threshold = 0.7
        self.accuracy_reward = 1.0

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['sticky_actions_counter'] = self.sticky_actions_counter
        return self.env.get_state(to_pickle)
    
    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['sticky_actions_counter']
        return from_pickle
        
    def reward(self, reward):
        """
        Modify the base reward based on the precision of long passes.
        Observing ball position change and calculating distance covered by a pass to check if it qualifies as a long pass.
        Additional rewards are given for accuracy.
        """
        components = {"base_score_reward": reward.copy(),
                      "long_pass_reward": [0.0] * len(reward)}

        observation = self.env.unwrapped.observation()
        if observation is None:
            return reward, components

        for rew_index in range(len(reward)):
            o = observation[rew_index]

            # Check if the current player just made a pass
            if o['sticky_actions'][8] == 1:  # Action 8 corresponds to 'action_pass'
                ball_start_pos = o['ball']
                next_observation, _, _, _ = self.env.step([0, 0])  # Assuming two players and no change in other's action
                ball_end_pos = next_observation[rew_index]['ball']
                
                # Calculate distance of the pass
                pass_distance = np.linalg.norm(ball_end_pos[:2] - ball_start_pos[:2])
                
                # Reward long and accurate passes
                if pass_distance > self.long_pass_threshold:
                    components["long_pass_reward"][rew_index] = self.accuracy_reward
                    reward[rew_index] += components["long_pass_reward"][rew_index]

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
