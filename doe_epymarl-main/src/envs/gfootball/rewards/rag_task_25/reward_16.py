import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that adds a dense reward for dribbling skills, specifically using 
    the Sprint action effectively and maintaining ball control under pressure.
    """

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['DribblingSkillRewardWrapper'] = self.sticky_actions_counter
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['DribblingSkillRewardWrapper']
        return from_pickle

    def reward(self, reward):
        """
        Augments the current reward based on effective use of Sprint and Dribble actions
        and maintaining control of the ball.
        """
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "dribble_sprint_bonus": [0.0] * len(reward)
        }
        
        if observation is None:
            return reward, components
        
        for rew_index in range(len(reward)):
            obs = observation[rew_index]

            # Check if player is dribbling and sprinting effectively
            if ('ball_owned_team' in obs and obs['ball_owned_team'] == 1 and
                'active' in obs and 'sticky_actions' in obs):
                
                dribbling = obs['sticky_actions'][9]  # action_dribble is index 9
                sprinting = obs['sticky_actions'][8]  # action_sprint is index 8

                if dribbling and sprinting:
                    components["dribble_sprint_bonus"][rew_index] += 0.05
                    reward[rew_index] += components["dribble_sprint_bonus"][rew_index]

                # Reward keeping control while being close to opponents
                player_pos = obs['right_team'][obs['active']]
                opponent_positions = obs['left_team']
                if np.any(np.linalg.norm(opponent_positions - player_pos, axis=1) < 0.1):
                    components["dribble_sprint_bonus"][rew_index] += 0.1
                    reward[rew_index] += components["dribble_sprint_bonus"][rew_index]

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
