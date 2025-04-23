import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that augments rewards based on mastering offensive strategies between midfielders and strikers.
    """
    
    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        # Rewards for control and aggressive offensive plays
        self.midfield_control_reward = 0.05
        self.striker_aggression_reward = 0.1
        self.coordinated_attack_bonus = 0.15

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper:sticky_actions'] = self.sticky_actions_counter
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle.get('CheckpointRewardWrapper:sticky_actions', np.zeros(10))
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "midfield_control_reward": [0.0] * len(reward),
                      "striker_aggression_reward": [0.0] * len(reward),
                      "coordinated_attack_bonus": [0.0] * len(reward)}

        if observation is None:
            return reward, components
            
        for rew_index, o in enumerate(observation):
            # Midfielders with the ball near the center advancing towards goal
            if o['active'] in o['midfielders'] and o['ball_owned_team'] == 0:
                mid_pos = o['right_team'][o['active']]
                ball_pos = o['ball'][:2]
                if np.linalg.norm(ball_pos - mid_pos) <= 0.1 and 0.3 < ball_pos[0] < 0.7:
                    reward[rew_index] += self.midfield_control_reward
                    components["midfield_control_reward"][rew_index] = self.midfield_control_reward
            
            # Strikers aggressively advancing towards the opponent's goal
            if o['active'] in o['strikers'] and o['ball_owned_team'] == 0:
                striker_pos = o['right_team'][o['active']]
                if striker_pos[0] > 0.8:
                    reward[rew_index] += self.striker_aggression_reward
                    components["striker_aggression_reward"][rew_index] = self.striker_aggression_reward
            
            # Extra bonus for coordinated plays: midfielders passing to strikers in attack position
            if o['active'] in o['midfielders'] and o['ball_owned_team'] == 0:
                mid_pos = o['right_team'][o['active']]
                if any(np.linalg.norm(mid_pos - o['right_team'][s]) < 0.2 for s in o['strikers']):
                    reward[rew_index] += self.coordinated_attack_bonus
                    components["coordinated_attack_bonus"][rew_index] = self.coordinated_attack_bonus

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
