import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that rewards agents specifically designed to enhance the capability of executing high passes with precision.
    This includes judging the trajectory, controlling the power, and recognizing when to use high passes effectively.
    """
    
    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.pass_quality_threshold = 0.7  # Arbitrary threshold for 'good' pass quality

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        # State specific to this wrapper should be restored here if needed
        return from_pickle

    def reward(self, reward):
        """
        Calculate rewards based on the agent's ability to execute high passes.
        
        The reward function focuses on:
        - Whether the pass was a high pass (based on `ball_direction` z-component).
        - Accuracy of the pass (considering proximity to theoretical best pass).
        - Rewarding only when ball gains possession by a teammate (using `ball_owned_team`).

        Parameters:
        - reward: List of float initial rewards from the underlying environment.

        Returns:
        - Updated reward as list of float, with added components for high passes.
        - A dict with information on the individual reward components.
        """
        observation = self.env.unwrapped.observation()
        components = {'base_score_reward': reward.copy(), 'pass_precision_reward': [0.0] * len(reward)}
        
        if observation is None:
            return reward, components

        for rew_index in range(len(reward)):
            o = observation[rew_index]

            # Evaluate the quality of high passes
            # Assuming fields 'ball_direction' and 'ball_owned_team' are used for determining pass quality
            if o['ball_owned_team'] in (0, 1):  # possession must belong to a team
                ball_direction = o['ball_direction']
                z_component = ball_direction[2]

                # Reward high Z direction as indication of high pass
                if z_component > self.pass_quality_threshold:
                    pass_quality_score = 1.0
                else:
                    pass_quality_score = 0.0

                components['pass_precision_reward'][rew_index] = pass_quality_score

                # Apply a reward for good high pass execution
                reward[rew_index] += 1.5 * components['pass_precision_reward'][rew_index]

        return reward, components

    def step(self, action):
        """
        Takes an action and returns the tuple (observation, reward, done, info).
        In addition to the environment's usual outputs, adds detailed reward component analysis.
        """
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
