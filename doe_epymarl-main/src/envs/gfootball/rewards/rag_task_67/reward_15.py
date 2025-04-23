import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a transitional skills reward with specific focus on passing and dribbling."""
    
    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.pass_completion_reward = 0.2
        self.dribble_skill_reward = 0.15
        self.skills_rewards_collected = {}

    def reset(self, **kwargs):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.skills_rewards_collected.clear()
        return super().reset(**kwargs)

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "pass_completion_reward": [0.0] * len(reward),
                      "dribble_skill_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        for i, obs in enumerate(observation):
            if obs['game_mode'] in [2, 3, 4, 5, 6]:  # Handling for various in-game modes like goal kicks, corner etc.
                continue

            player_has_ball = (obs['ball_owned_team'] == 1) and (obs['ball_owned_player'] == obs['active'])
            
            # Check if the agent’s team has ball control
            if player_has_ball:
                last_action = obs['sticky_actions'][-1]  # the last action played
                
                # Detect short or long pass completion
                if last_action in [2, 3]:  # indices for short and long passes
                    components['pass_completion_reward'][i] = self.pass_completion_reward
                # Reward dribbling action
                elif last_action == 9:  # index for dribble action
                    components['dribble_skill_reward'][i] = self.dribble_skill_reward
                
            # Aggregating rewards for respective skill
            reward[i] += (components['pass_completion_reward'][i] + components['dribble_skill_reward'][i])
        
        return reward, components
        
    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        return observation, reward, done, info

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = self.skills_rewards_collected
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.skills_rewards_collected = from_pickle['CheckpointRewardWrapper']
        return from_pickle
