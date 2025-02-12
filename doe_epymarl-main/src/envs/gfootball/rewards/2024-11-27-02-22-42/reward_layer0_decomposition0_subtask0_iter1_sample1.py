import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that enhances the reward signal for offensive soccer strategies."""
    
    def __init__(self, env):
        super().__init__(env)
        # Define reward adjustments for specific actions
        self.pass_reward = 0.5
        self.shot_reward = 1.0
        self.dribble_reward = 0.3

    def reset(self):
        # Reset the environment to its initial state
        return self.env.reset()

    def get_state(self, to_pickle):
        # Get the state of the environment for serialization purposes
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        # Set the state of the environment from the deserialized object
        return self.env.set_state(state)

    def reward(self, reward):
        """Enhance the reward based on offensive actions: pass, shot and dribble."""
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": np.copy(reward),  # retain original reward
                      "pass_reward": np.zeros(len(reward)),
                      "shot_reward": np.zeros(len(reward)),
                      "dribble_reward": np.zeros(len(reward))}

        for index, obs in enumerate(observation):
            # Check if the action is pass (short, long, high with ids in example: 1, 9 and hypothetically 8)
            if obs['sticky_actions'][1] or obs['sticky_actions'][9] or \
               (len(obs['sticky_actions']) > 8 and obs['sticky_actions'][8]):
                components["pass_reward"][index] += self.pass_reward
                reward[index] += self.pass_reward

            # Check if the action is shot; typically represented by game_mode==Shot(3)
            if obs['game_mode'] == 3 and obs['ball_owned_player'] == obs['active']:
                components["shot_reward"][index] += self.shot_reward
                reward[index] += self.shot_reward
            
            # Check if the action is dribble (assumed index 10 here, check index specifics for dribble)
            if len(obs['sticky_actions']) > 10 and obs['sticky_actions'][10]:
                components["dribble_reward"][index] += self.dribble_reward
                reward[index] += self.dribble_reward

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        # Enhance the reward using the defined function
        reward, components = self.reward(reward)
        
        # Add the final reward and components to info for further use
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        
        return observation, reward, done, info
