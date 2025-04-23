import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper to increase defensive capabilities in handling direct attacks,
    focusing on positioning and quick counterattacks.
    """
    
    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)  # Initialize the sticky actions count.
        self.near_goal_provocations = 0
        
    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.near_goal_provocations = 0
        return self.env.reset()
    
    def get_state(self, to_pickle):
        to_pickle['sticky_actions_counter'] = self.sticky_actions_counter
        return self.env.get_state(to_pickle)
    
    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle.get('sticky_actions_counter', np.zeros(10, dtype=int))
        return from_pickle
    
    def reward(self, reward):
        observations = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "defensive_positioning": [0.0] * len(reward),
                      "counterattack_speed": [0.0] * len(reward)}
        
        for i, o in enumerate(observations):
            ball_owned_team = o.get('ball_owned_team', -1)
            if ball_owned_team == 1:  # Assuming the agent's team is team 1 (right-side defense).
                ball_pos_x = o['ball'][0]
                
                # Encourage defensive positioning near their own goal zone.
                if -0.8 <= ball_pos_x <= -0.5:
                    components["defensive_positioning"][i] += 0.1  # Adding a small reward for positioning.
                
                # Encourage quick shifting to offense after getting possession.
                if (ball_owned_team == 0 and o['right_team_active'][o['active']]) and (o['right_team_direction'][o['active']][0] > 0):
                    components["counterattack_speed"][i] += 0.1  # Increased speed towards opponent's goal. 

        # Sum the extra components into the provided base reward for each agent.
        total_rewards = [sum(x) for x in zip(reward, components["defensive_positioning"], components["counterattack_speed"])]
        return total_rewards, components
    
    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        
        # Update the sticky actions counter and add to the info object
        obs = self.env.unwrapped.observation()
        for key, val in components.items():
            info[f"component_{key}"] = sum(val)
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                info[f"sticky_actions_{i}"] = action
                
        return obs, reward, done, info
