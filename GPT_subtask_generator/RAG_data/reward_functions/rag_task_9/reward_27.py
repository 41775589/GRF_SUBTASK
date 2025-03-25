import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense reward for offensive skills."""
    
    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        # Initialize sticky actions counter for detailed info on agent's behavior
        self.sticky_actions_counter = np.zeros(10, dtype=int)

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
        # Capturing the environment's observations for reward adjustments
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy()}
        
        if observation is None:
            return reward, components

        # Extract specific game-mode related rewards
        for idx, o in enumerate(observation):
            # Check if the game is in normal play mode
            if o['game_mode'] == 0:
                if 'active' in o and o['ball_owned_player'] == o['active'] and o['ball_owned_team'] == 0:
                    if 'sticky_actions' in o:
                        actions = o['sticky_actions']
                        # Rewards related to passing, shooting, dribbling, and sprinting
                        pass_reward = actions[1] + actions[2]  # short pass + long pass
                        shot_reward = actions[3]  # shot
                        dribble_reward = actions[9]  # dribble
                        sprint_reward = actions[8]  # sprint

                        # Accumulate rewards based on specific actions
                        components[f"pass_reward_agent_{idx}"] = 0.1 * pass_reward
                        components[f"shot_reward_agent_{idx}"] = 0.2 * shot_reward
                        components[f"dribble_reward_agent_{idx}"] = 0.1 * dribble_reward
                        components[f"sprint_reward_agent_{idx}"] = 0.05 * sprint_reward

                        # Multiply rewards by respective coefficients and add them up
                        reward[idx] += (components[f"pass_reward_agent_{idx}"] +
                                        components[f"shot_reward_agent_{idx}"] +
                                        components[f"dribble_reward_agent_{idx}"] +
                                        components[f"sprint_reward_agent_{idx}"])

        return reward, components

    def step(self, action):
        # Perform the action on the environment
        observation, reward, done, info = self.env.step(action)
        
        # Update the reward based on custom function
        reward, components = self.reward(reward)
        
        # Update info dict with detailed rewards and final compounded reward
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        
        # Gather observations for the current state
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        
        # Update sticky actions info
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                info[f"sticky_actions_{i}"] = action

        return observation, reward, done, info
