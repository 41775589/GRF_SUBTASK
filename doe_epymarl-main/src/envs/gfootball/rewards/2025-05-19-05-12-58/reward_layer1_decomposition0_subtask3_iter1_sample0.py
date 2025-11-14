import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A reward wrapper to enhance learning for strategic passing, utilizing Short Pass (action 8) and High Pass (action 9).
    This helps in creating scoring opportunities by manipulating defensive setups.
    """

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)  # Tracking sticky actions for debug purposes.

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)  # Reset the counter on environment reset.
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {}  # Placeholder for any state specifics from rewards.
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)  # Load state from the environment's previous state.
        # Load specific state alterations from CheckpointRewardWrapper if any were saved.
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()  # Get current observation
        if observation is None:
            return reward, {}

        components = {
            "base_score_reward": reward.copy(),  # Baseline reward from the environment.
            "pass_quality_reward": [0.0] * len(reward)  # Reward for strategic passing.
        }

        for index in range(len(reward)):
            o = observation[index]
            ball_owned_team = o['ball_owned_team']  # Who owns the ball

            # Check if the active player is executing a pass
            passing_action = o['sticky_actions'][8] or o['sticky_actions'][9]

            if ball_owned_team == 1 and passing_action:
                # More sophisticated calculation can be added here based on positioning, etc.
                # Currently gives additional reward for performing the passing action
                components["pass_quality_reward"][index] = 0.2  # Incremental reward for passing
                reward[index] += components["pass_quality_reward"][index]

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        
        # Add additional details about rewards and actions to info for debugging
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                self.sticky_actions_counter[i] = (action == 1)
                info[f"sticky_actions_{i}"] = self.sticky_actions_counter[i]
                
        return observation, reward, done, info
