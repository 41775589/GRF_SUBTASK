import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a structured reward based on offensive strategy elements:
    accurate shooting, effective dribbling, and diverse pass types."""
    
    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)
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
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "shooting_accuracy_bonus": [0.0] * len(reward),
                      "dribble_bonus": [0.0] * len(reward),
                      "pass_bonus": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            components["base_score_reward"][rew_index] = reward[rew_index]  # Preserve the original reward
            
            # Bonus for shooting accuracy towards the goal when in possession and closer to opponent's goal
            if o['ball_owned_team'] == 0 and o['active'] == o['ball_owned_player']:
                if o['ball'][0] > 0.5:  # assuming ball x-coords >0.5 is near opponent's goal in a normalized football field
                    components["shooting_accuracy_bonus"][rew_index] = 0.1
                    reward[rew_index] += components["shooting_accuracy_bonus"][rew_index]
            
            # Dribbling effectiveness: bonus for maintaining ball possession under opponent's pressure
            if o['ball_owned_team'] == 0 and  o['active'] == o['ball_owned_player'] and o['sticky_actions'][9]:  # action_dribble index
                opponent_closeness = [np.linalg.norm(o['ball'] - opp_pos) for opp_pos in o['right_team']]
                if min(opponent_closeness) < 0.1:  # if any opponent is within a threshold distance
                    components["dribble_bonus"][rew_index] = 0.2
                    reward[rew_index] += components["dribble_bonus"][rew_index]

            # Passing efficiency: reward based on pass actions leading to change in ball ownership within the team
            if o['ball_owned_team'] == 0 and o['sticky_actions'][0] or o['sticky_actions'][5]:  # action_left or action_bottom_right for passing
                components["pass_bonus"][rew_index] = 0.05
                reward[rew_index] += components["pass_bonus"][rew_index]

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
