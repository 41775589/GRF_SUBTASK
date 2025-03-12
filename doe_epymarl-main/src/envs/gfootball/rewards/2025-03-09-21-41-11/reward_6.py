import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense reward scheme focused on offensive football strategies,
    including accurate shooting, effective dribbling, and various pass types."""
    
    def __init__(self, env):
        super().__init__(env)
        self.pass_success_reward = 0.05
        self.shoot_accuracy_reward = 0.1
        self.dribble_efficiency_reward = 0.03
    
    def reset(self):
        """Resets the environment and any internal variables."""
        return self.env.reset()
    
    def get_state(self, to_pickle):
        """Save the state as part of a pickle."""
        return self.env.get_state(to_pickle)
    
    def set_state(self, state):
        """Set the state from unpickled data."""
        return self.env.set_state(state)
    
    def reward(self, reward):
        """Enhance the base reward given by the environment based on offensive moves."""
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "pass_success_reward": [0.0] * len(reward),
                      "shoot_accuracy_reward": [0.0] * len(reward),
                      "dribble_efficiency_reward": [0.0] * len(reward)}
        
        if observation is None:
            return reward
        
        assert len(reward) == len(observation)

        for idx in range(len(reward)):
            o = observation[idx]
            if o['game_mode'] != 0:  # Consideration only during normal play
                continue

            ball_owned_team = o['ball_owned_team']
            controlled_player_team = 0 if o['active'] in o['left_team_roles'] else 1
            
            # Check if controlled player has ball
            if ball_owned_team == controlled_player_team:
                # Applying rewards for successful passes
                if o['sticky_actions'][9] > 0:  # Pass action
                    components["pass_success_reward"][idx] += self.pass_success_reward
                    reward[idx] += components["pass_success_reward"][idx]
                
                # Applying rewards for effective dribbling
                if o['sticky_actions'][8] > 0:  # Dribble action
                    components["dribble_efficiency_reward"][idx] += self.dribble_efficiency_reward
                    reward[idx] += components["dribble_efficiency_reward"][idx]

                # Applying rewards for on-target shots
                goal_position = [1, 0] if controlled_player_team == 0 else [-1, 0]
                distance_to_goal = np.linalg.norm(np.array(goal_position) - np.array(o['ball']))
                if o['sticky_actions'][0] > 0 and distance_to_goal < 0.3:  # Shot action close to goal
                    components["shoot_accuracy_reward"][idx] += self.shoot_accuracy_reward
                    reward[idx] += components["shoot_accuracy_reward"][idx]

        return reward, components

    def step(self, action):
        """
        Step through an environment using the specified action.
        """
        observation, reward, done, info = self.env.step(action)
        modified_reward, components = self.reward(reward)
        
        # Sum the component values and store in the info dictionary
        info['modified_reward'] = sum(modified_reward)
        for key, component in components.items():
            info[f"component_{key}"] = sum(component)
            
        return observation, modified_reward, done, info
