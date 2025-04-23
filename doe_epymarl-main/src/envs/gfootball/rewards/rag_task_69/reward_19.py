import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that adds a dense reward for offensive actions like shooting accuracy,
    dribbling to evade opponents, and executing strategic passes.
    """
    
    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.shooting_accuracy_reward = 0.2
        self.dribbling_skill_reward = 0.15
        self.pass_effectiveness_reward = 0.1

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {
            'sticky_actions_counter': self.sticky_actions_counter
        }
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['CheckpointRewardWrapper']['sticky_actions_counter']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "shooting_accuracy": [0.0] * len(reward),
            "dribbling_skill": [0.0] * len(reward),
            "pass_effectiveness": [0.0] * len(reward)
        }
        
        if observation is None:
            return reward, components

        for agent_index, o in enumerate(observation):
            # Base score based on game outcome
            base_reward = reward[agent_index]
            
            # Shooting accuracy: reward if player shoots near the goal
            if 'ball_direction' in o and 'ball' in o:
                goal_y_range = [-0.044, 0.044]
                if o['ball_owned_team'] == 0 and np.abs(o['ball'][1]) in goal_y_range and o['ball'][0] > 0.7:
                    components["shooting_accuracy"][agent_index] += self.shooting_accuracy_reward

            # Dribbling skill: reward successful dribble based on sticky_actions 'dribble' is active and player evades an opponent
            if 'ball_owned_team' in o and o['ball_owned_team'] == 0:
                if o['sticky_actions'][9] == 1:  # dribbling action
                    components["dribbling_skill"][agent_index] += self.dribbling_skill_reward
            
            # Pass effectiveness: reward successful long or high passes
            # (simulation of measuring pass types, assumes a received active state change)
            previous_active = self.sticky_actions_counter
            current_active = o['active']
            if previous_active[agent_index] != current_active:
                components["pass_effectiveness"][agent_index] += self.pass_effectiveness_reward
                
            # Aggregate total reward for this agent
            reward[agent_index] = (
                base_reward +
                components["shooting_accuracy"][agent_index] +
                components["dribbling_skill"][agent_index] +
                components["pass_effectiveness"][agent_index]
            )

            # Update sticky actions counter for next step
            self.sticky_actions_counter = o['sticky_actions']

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
