import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense reward for 'Shoot' based on accuracy and shot power, 
    fostering mastery of scoring opportunities."""
    
    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.shot_threshold = 0.5  # Threshold representing distance from the goal to consider a 'good' shot taken
        self.pressure_modifier = 0.1  # Reward modifier that increases value of shots under pressure
        self.reset()

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['sticky_actions_counter'] = self.sticky_actions_counter
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        state = self.env.set_state(state)
        self.sticky_actions_counter = state.get('sticky_actions_counter', np.zeros(10, dtype=int))
        return state

    def reward(self, reward):
        observation = self.unwrapped.observation()
        components = {"base_score_reward": reward.copy()}
        
        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            is_shooting_action = o['sticky_actions'][9]  # Assuming index 9 corresponds to 'Shoot' action
            
            if is_shooting_action:
                ball_pos = o['ball'][0]
                pressure = len([player for player in o['right_team'] if np.linalg.norm(player - o['ball'][:2]) < 0.2])
                
                # If shot is taken towards the opponent's goal and the ball is beyond a specific threshold closer to opponent's end
                if ball_pos > self.shot_threshold:
                    components[f'shoot_accuracy_reward_{rew_index}'] = 1.0 + self.pressure_modifier * pressure
                else:
                    components[f'shoot_accuracy_reward_{rew_index}'] = 0.5
                    
                reward[rew_index] += components[f'shoot_accuracy_reward_{rew_index}']
            else:
                components[f'shoot_accuracy_reward_{rew_index}'] = 0

        self.sticky_actions_counter.fill(0)
        for agent_obs in observation:
            for i, action in enumerate(agent_obs['sticky_actions']):
                if action == 1:
                    self.sticky_actions_counter[i] += 1
                    
        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = value
        return observation, reward, done, info
