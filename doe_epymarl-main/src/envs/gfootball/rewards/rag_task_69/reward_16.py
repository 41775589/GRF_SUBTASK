import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A reward wrapper that encourages specific offensive strategies in the Google Research Football environment."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        """ Reset the sticky actions counter at the start of each episode."""
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        """Save the current state of the environment and wrapper."""
        to_pickle['sticky_actions_counter'] = self.sticky_actions_counter
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """Restore the state of the environment and wrapper."""
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle.get('sticky_actions_counter', np.zeros(10, dtype=int))
        return from_pickle

    def reward(self, reward):
        """ Customize the reward function to promote accurate shooting, dribbling, and strategic passing."""
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy()}
        
        if observation is None or len(observation) == 0:
            return reward, components
        
        # Initialize reward component arrays
        components['shooting_reward'] = [0.0] * len(reward)
        components['dribbling_reward'] = [0.0] * len(reward)
        components['passing_reward'] = [0.0] * len(reward)
        
        for index, obs in enumerate(observation):
            if obs is None:
                continue
            
            # Calculate shooting reward
            if obs['ball_owned_team'] == 0 and obs['game_mode'] in {0, 1}:  # Normal play or kick-off
                goal_distance = np.abs(obs['ball'][0] - 1)  # distance from opponent's goal
                if goal_distance < 0.3:  # close to opponent's goal
                    components['shooting_reward'][index] += 0.5
            
            # Calculate dribbling reward
            if obs['sticky_actions'][9] == 1: # dribbling action active
                components['dribbling_reward'][index] += 0.1
            
            # Calculate passing reward
            if obs['game_mode'] == 2 or obs['game_mode'] == 4:  # Goal kick or Corner
                components['passing_reward'][index] += 0.3
            
            # Accumulate individual components
            final_reward = (components['base_score_reward'][index] + 
                            components['shooting_reward'][index] +
                            components['dribbling_reward'][index] +
                            components['passing_reward'][index])
            
            reward[index] = final_reward

        return reward, components

    def step(self, action):
        """Step function processes agent actions, collects the reward, and updates game state."""
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
