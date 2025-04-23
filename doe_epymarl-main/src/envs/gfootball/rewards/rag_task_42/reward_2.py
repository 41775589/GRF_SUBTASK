import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds rewards based on midfield dynamics and strategic positioning for offense and defense transitions."""
    
    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.midfield_checkpoints = []
        self.defensive_repositioning = []
        self.offensive_transitions = []
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.midfield_checkpoints = []
        self.defensive_repositioning = []
        self.offensive_transitions = []
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['midfield_checkpoints'] = self.midfield_checkpoints
        to_pickle['defensive_repositioning'] = self.defensive_repositioning
        to_pickle['offensive_transitions'] = self.offensive_transitions
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.midfield_checkpoints = from_pickle['midfield_checkpoints']
        self.defensive_repositioning = from_pickle['defensive_repositioning']
        self.offensive_transitions = from_pickle['offensive_transitions']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "checkpoint_reward": [0.0] * len(reward),
            "transition_reward": [0.0] * len(reward)
        }
        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            
            midfield_position_x = 0.0  # Approximately midfield x positional value
            
            # Reward for maintaining positions in the midfield area and transitioning
            if -0.2 <= o['ball'][0] <= 0.2:
                self.midfield_checkpoints.append(0.1)
                components["checkpoint_reward"][rew_index] = 0.1
            
            # Strategically reward defensive repositioning when ball is with opponent
            if o['ball_owned_team'] == 1:  # Ball owned by opponent
                teams_distance = np.linalg.norm(o['left_team'] - o['right_team'][o['ball_owned_player']])
                if teams_distance < 0.3:  # Close to the ball owned player
                    self.defensive_repositioning.append(0.2)
                    components["transition_reward"][rew_index] += 0.2

            # Reward offensive transitions when ball possession changes
            if self.prev_ball_owned != o['ball_owned_team']:
                if o['ball_owned_team'] == 0:  # Ball gained by our team
                    self.offensive_transitions.append(0.3)
                    components["transition_reward"][rew_index] += 0.3
            
            # Update the rewards list
            reward[rew_index] += components["checkpoint_reward"][rew_index]
            reward[rew_index] += components["transition_reward"][rew_index]
            
            # Save the previous ball ownership state
            self.prev_ball_owned = o['ball_owned_team']

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
            for i, action in enumerate(agent_obs['sticky_actions']):
                info[f"sticky_actions_{i}"] = action
        return observation, reward, done, info
