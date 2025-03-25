import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that modifies the reward based on the scenario where quick decision-making 
    and ball handling after recovering possession are crucial for counter-attacks.
    """
    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.ball_lost_position = None  # Track the position where the ball was last lost
        self.counter_attack_mode = False  # Indicates if a counter-attack could potentially be initiated
        self.counter_attack_bonus = 0.5  # Bonus reward for successful quick counter-attacks

    def reset(self):
        """ Resets the environment and clears counter attack flags and positions """
        self.ball_lost_position = None
        self.counter_attack_mode = False
        return self.env.reset()

    def reward(self, reward):
        """
        Modify the reward for actions that contribute to quick, effective counter-attacks.

        Args:
            reward (float): the original reward given by the environment
        
        Returns:
            Modified reward reflecting quick ball recovery and progression.
        """
        observation = self.env.unwrapped.observation()
        
        if observation is None:
            return reward
        
        # Components to store reward additions such as base reward and counter-attack reward
        components = {"base_score_reward": reward, "counter_attack_reward": 0}

        current_ball_owned_team = observation['ball_owned_team']
        ball_position = observation['ball']

        # Detect ball possession recovery (0 represents the controlled left team)
        if current_ball_owned_team == 0:
            if self.ball_lost_position is not None:
                # Check if recovery has led to a quick move towards the opponent's side
                if ball_position[0] > self.ball_lost_position[0] + 0.2:  # Assuming positive x-direction is towards opponent
                    reward += self.counter_attack_bonus
                    components["counter_attack_reward"] = self.counter_attack_bonus
                # Reset counter-attack potential after reward is given
                self.ball_lost_position = None
                self.counter_attack_mode = False
            else:
                if not self.counter_attack_mode:
                    # If the ball is previously owned by no one or the opponent, consider next phase for counter
                    self.counter_attack_mode = True
            
        elif current_ball_owned_team != 0:
            # If ball is lost, store the ball position
            self.ball_lost_position = ball_position
            self.counter_attack_mode = False
        
        return reward, components

    def step(self, action):
        """ Execute an environment step, update observation, components and reward. """
        observation, reward, done, info = self.env.step(action)
        modified_reward, components = self.reward(reward)

        # Update info dictionary with rewards breakdown and total reward
        info["final_reward"] = modified_reward
        for key, value in components.items():
            info[f"component_{key}"] = value

        return observation, modified_reward, done, info
