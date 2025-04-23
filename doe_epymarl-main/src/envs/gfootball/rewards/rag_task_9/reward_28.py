import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that modifies the reward function to foster offensive football skills."""
    
    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper_stickies'] = self.sticky_actions_counter
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle.get('CheckpointRewardWrapper_stickies', np.zeros(10, dtype=int))
        return from_pickle
    
    def reward(self, reward):
        """Augment the reward based on actions related to offensive skills: dribbling, passing, and shooting."""
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(), "offensive_skill_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components
        
        assert len(reward) == len(observation)

        # Reward multipliers
        pass_reward = 0.01  # Reward for executing any type of pass
        shot_reward = 0.05  # Reward for taking a shot
        dribble_reward = 0.02  # Reward for dribbling action
        sprint_reward = 0.01  # Reward for sprint action

        for idx, o in enumerate(observation):
            current_sticky_actions = o.get('sticky_actions', [0] * 10)
            
            # Check if the ball is owned by an active player of own team
            if o['ball_owned_team'] == 1 and o['active'] == o['ball_owned_player']:
                # Check rewards for passing
                if current_sticky_actions[0] or current_sticky_actions[1]:
                    components["offensive_skill_reward"][idx] += pass_reward
                
                # Check reward for shooting
                if o['game_mode'] in [2, 6]:
                    # Penalize passing in shot or penalty mode, but reward the shot attempt
                    components["offensive_skill_reward"][idx] += shot_reward
                
                # Check reward for dribbling
                if current_sticky_actions[9]:
                    components["offensive_skill_reward"][idx] += dribble_reward

                # Check reward for sprinting
                if current_sticky_actions[8]:
                    components["offensive_skill_reward"][idx] += sprint_reward

            # Apply calculated reward components
            reward[idx] += 2 * components["offensive_skill_reward"][idx]  # Scaling the reward to make it significant
        
        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)

        # Aggregate and pass detailed reward information
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)

        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                self.sticky_actions_counter[i] += action
                info[f"sticky_actions_{i}"] = action

        return observation, reward, done, info
