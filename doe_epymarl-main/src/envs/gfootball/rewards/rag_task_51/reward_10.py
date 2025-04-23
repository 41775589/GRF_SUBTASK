import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A reward wrapper that focuses on specialized goalkeeper training. It rewards the agent
    for shot-stopping, showing quick reflexes (reacting to rapid changes in ball direction),
    and initiating counter-attacks with accurate passes. This encourages the goalkeeper to
    both defend effectively and contribute to the offense.
    """
    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.ball_direction_change_counter = 0
        self.previous_ball_direction = None

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.ball_direction_change_counter = 0
        self.previous_ball_direction = None
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {
            'ball_direction_change_counter': self.ball_direction_change_counter,
            'previous_ball_direction': self.previous_ball_direction}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.ball_direction_change_counter = from_pickle['CheckpointRewardWrapper']['ball_direction_change_counter']
        self.previous_ball_direction = from_pickle['CheckpointRewardWrapper']['previous_ball_direction']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(), "shot_stopping": [0.0, 0.0], 
                      "quick_reflex": [0.0, 0.0], "counter_attack_initiation": [0.0, 0.0]}
        
        if observation is None:
            return reward, components
        
        assert len(reward) == len(observation)
        
        for i, o in enumerate(observation):
            if o['ball_owned_team'] == 0 and o['ball_owned_player'] == o['active']:
                my_role = o['left_team_roles'][o['active']]
                
                # Reward goalkeeper for shot stopping
                if my_role == 0 and o['game_mode'] == 6:  # Penalty mode
                    components["shot_stopping"][i] = 1.0
                    reward[i] += components["shot_stopping"][i]
                
                # Ball direction change detection for quick reflexes
                current_ball_direction = o['ball_direction']
                if self.previous_ball_direction is not None:
                    direction_change = np.linalg.norm(current_ball_direction - self.previous_ball_direction)
                    components["quick_reflex"][i] = direction_change
                    reward[i] += components["quick_reflex"][i]
                self.previous_ball_direction = current_ball_direction

                # Reward goalkeeper for initiating counter-attacks
                if reward[i] == 1 and o['designated'] == o['ball_owned_player']:
                    components["counter_attack_initiation"][i] = 0.5
                    reward[i] += components["counter_attack_initiation"][i]
        
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
            for j, action in enumerate(agent_obs['sticky_actions']):
                info[f"sticky_actions_{j}"] = action
        return observation, reward, done, info
