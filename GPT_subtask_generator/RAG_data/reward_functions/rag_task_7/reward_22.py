import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that adds a dense reward based on the effectiveness and timeliness
    of sliding tackles in defensive maneuvers under high-pressure situations.
    It focuses on the sliding tackle action accuracy.
    """

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)  # to keep count of actions executed
    
    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['actions_performed'] = self.sticky_actions_counter
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle.get('actions_performed', np.zeros(10, dtype=int))
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "tackle_accuracy_reward": [0.0] * len(reward)}

        # Update only if the observations are valid
        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]

            # First, check if the game mode allows/considers tackles and manipulation
            if o['game_mode'] not in [0, 3, 4, 6]:  # Only normal, freekicks, corners, or penalty modes
                continue

            # Check if the active player is on the side making a defensive play
            if o['ball_owned_team'] == 1:
                continue

            # Sliding tackle specific logic:
            # We assume a sliding tackle action id in sticky_actions (e.g., id 9)
            tackle_action_id = 9
            if o['sticky_actions'][tackle_action_id]:

                # Check how close the ball is when tackling if the opponent has the ball
                if o['ball_owned_team'] == 1:  # Enemy team has the ball
                    player_pos = o['right_team'][o['active']]
                    opponent_pos = o['left_team'][o['ball_owned_player']]
                    distance = np.linalg.norm(player_pos - opponent_pos)

                    # Reward for tackling at the right moment and distance
                    if distance < 0.1: # arbitrary threshold for effective tackle distance
                        components["tackle_accuracy_reward"][rew_index] = 0.5  # higher reward for effective defense
                    else:
                        components["tackle_accuracy_reward"][rew_index] = -0.1  # small penalty for mistimed tackle

                reward[rew_index] += components["tackle_accuracy_reward"][rew_index]
        
        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)

        # Update the sticky actions counter for analysis or other purposes
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                info[f"sticky_actions_{i}"] = action

        # Append component wise rewards to info for better monitoring
        info['final_reward'] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)

        return observation, reward, done, info
