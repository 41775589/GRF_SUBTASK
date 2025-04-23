import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense checkpoint reward for defensive play and counterattacks."""

    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        base_reward = reward.copy()
        defensive_reward = [0.0] * len(reward)
        
        if observation is None:
            return reward, {'base_score_reward': reward, 'defensive_reward': defensive_reward}

        for i in range(len(reward)):
            o = observation[i]
            # Encourage maintaining possession and moving forward defensively
            if o['ball_owned_team'] == 1:  # Assuming team 1 is the defensive team
                # Player has the ball
                if o['ball_owned_player'] == o['active']:
                    forward_movement = o['right_team_direction'][o['active']][0]  # x-direction movement
                    positioning = o['right_team'][o['active']][0]  # x-position of the player
                    
                    # Reward based on moving the ball forward and positioning close to goal
                    if forward_movement > 0 and positioning < 0.5:
                        defensive_reward[i] = 0.1  # Reward for moving forward in a controlled manner

                    # Reward for clearing the ball under pressure
                    if o['game_mode'] in [2, 3, 4, 5]:  # Pressure situations: goal kick, free kick, corner, throw in
                        defensive_reward[i] = 0.2

        # Aggregate the rewards and reward components
        for i in range(len(reward)):
            reward[i] += defensive_reward[i]
        
        return reward, {'base_score_reward': base_reward, 'defensive_reward': defensive_reward}

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
