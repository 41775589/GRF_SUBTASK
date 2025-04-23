import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that adds custom rewards based on strategic positioning, lateral and backward movement,
    and acceleration criteria to enhance the team's defensive resilience and counterattack capability.
    """
    
    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
    
    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()
    
    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {}
        return self.env.get_state(to_pickle)
    
    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        # Load any necessary data if `CheckpointRewardWrapper` state was saved.
        return from_pickle
    
    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy()}
        
        if observation is None:
            return reward, components
        
        assert len(reward) == len(observation)
        
        for rew_index in range(len(reward)):
            o = observation[rew_index]
            # Implement your strategic positioning and movement rewards conditioning here.
            
            # Encourage lateral and backward movement when defending.
            if o['ball_owned_team'] != -1 and o['ball_owned_team'] != o['active']:
                # Encourage defending players to position themselves between the ball carrier and the goal.
                goal_y = 0.0
                player_pos = o['left_team'][o['active']] if o['ball_owned_team'] == 1 else o['right_team'][o['active']]
                ball_pos = o['ball']
                
                # Normalize coordinates considering playing field dimensions.
                # Goal is at x = 1 for right team and x = -1 for left team.
                goal_distance = np.abs(goal_y - player_pos[1])
                ball_distance = np.linalg.norm(player_pos - ball_pos[:2])
                
                # Incentivize players to move laterally (across y-axis) to cover more field.
                lateral_movement = np.abs(player_pos[1] - ball_pos[1])
                components['defensive_positioning'] = 0.1 * (lateral_movement / goal_distance)
                
                # Promote quick reaction to ball possession changes for counterattacks.
                if o['ball_owned_team'] != self.env.unwrapped.prev_ball_owned_team:
                    components['quick_reaction_bonus'] = 0.5
                else:
                    components['quick_reaction_bonus'] = 0.0
                
                # Adding/Subtracting the calculated rewards or penalties.
                reward[rew_index] += components['defensive_positioning'][rew_index]
                reward[rew_index] += components['quick_reaction_bonus'][rew_index]
        
        return reward, components
    
    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = np.sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = np.sum(value)
        
        # Track sticky actions for analytics.
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                self.sticky_actions_counter[i] += action
                
        return observation, reward, done, info
