import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adjusts rewards to better optimize defensive football skills and address identified issues."""
    
    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.positional_defense_reward_factor = 0.05
        self.action_based_defense_reward_factor = 0.3
    
    def reset(self):
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        base_score_reward = reward.copy()
        positional_defense_reward = [0.0] * len(reward)
        action_based_defense_reward = [0.0] * len(reward)

        if observation is None:
            return reward, {'base_score_reward': base_score_reward, 'positional_defense_reward': positional_defense_reward, 'action_based_defense_reward': action_based_defense_reward}

        for rew_index, o in enumerate(observation):
            # Including defensive rewards based on opponent's proximity to the goal
            own_goal_position = [-1, 0]  # Goal position for the team that must defend
            opponent_positions = o['right_team']  # Assuming own team is the left team
            min_distance = np.min([np.linalg.norm(player - own_goal_position) for player in opponent_positions])
            positional_defense_reward[rew_index] = self.positional_defense_reward_factor / (min_distance + 0.1)
            
            # Evaluating specific defensive actions (Slide, Sprint, Stop Sprint, Short Pass)
            if 'sticky_actions' in o:
                action_rewards = o['sticky_actions'][0] * 0.5 + o['sticky_actions'][3] + o['sticky_actions'][7] * 0.3 + o['sticky_actions'][1] * 0.2
                action_based_defense_reward[rew_index] = self.action_based_defense_reward_factor * action_rewards

            # Update reward considering additional contributions
            reward[rew_index] = base_score_reward[rew_index] + positional_defense_reward[rew_index] + action_based_defense_reward[rew_index]

        return reward, {
            'base_score_reward': base_score_reward, 
            'positional_defense_reward': positional_defense_reward,
            'action_based_defense_reward': action_based_defense_reward
        }

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        return observation, reward, done, info
