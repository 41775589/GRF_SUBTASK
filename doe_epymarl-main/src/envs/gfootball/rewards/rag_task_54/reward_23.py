import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that enhances the effectiveness of collaborative plays between shooters and passers to fully exploit scoring opportunities.
    """

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.passing_reward = 0.05
        self.shooting_reward = 0.1
        self.teammate_interaction_reward = 0.2

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {
            'sticky_actions_counter': self.sticky_actions_counter,
        }
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['CheckpointRewardWrapper'].get('sticky_actions_counter', np.zeros(10, dtype=int))
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        
        components = {
            "base_score_reward": reward.copy(),
            "passing_reward": [0.0] * len(reward),
            "shooting_reward": [0.0] * len(reward),
            "teammate_interaction_reward": [0.0] * len(reward)
        }

        if observation is None:
            return reward, components
        
        assert len(reward) == len(observation)

        for rew_index, o in enumerate(observation):
            if o['ball_owned_team'] == 0:
                last_player = o['ball_owned_player']
                if o['game_mode'] in {2, 4, 7}:  # Modes corresponding to GoalKick, Corner, Penalty
                    if o['active'] == last_player:
                        components['shooting_reward'][rew_index] += self.shooting_reward
                    
                elif o['game_mode'] == 3:  # FreeKick, typically a shooting opportunity
                    if o['active'] == last_player:
                        components['shooting_reward'][rew_index] += self.shooting_reward * 2
            
            # Check interactions with teammates
            if 'pass_count' in o:
                if o['pass_count'] > 0:
                    components['teammate_interaction_reward'][rew_index] = self.teammate_interaction_reward * o['pass_count']
                    components['passing_reward'][rew_index] = self.passing_reward * o['pass_count']
            
            # Calculate final reward for this agent
            reward[rew_index] += components['passing_reward'][rew_index] + \
                                 components['shooting_reward'][rew_index] + \
                                 components['teammate_interaction_reward'][rew_index]

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
                self.sticky_actions_counter[i] += action
                info[f"sticky_actions_{i}"] = self.sticky_actions_counter[i]
        
        return observation, reward, done, info
