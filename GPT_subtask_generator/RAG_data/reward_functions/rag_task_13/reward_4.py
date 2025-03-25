import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper focused on reinforcing defensive actions like marking, blocking, and 
    contesting forward moves by the opponent."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        # Define additional attributes
        self.blocking_bonus = 0.1
        self.marking_bonus = 0.05
        self.defensive_positions = {'left_back', 'right_back', 'centre_back'}

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        return self.env.set_state(state)

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "defensive_bonuses": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            components["defensive_bonuses"][rew_index] = 0.0

            # Task-specific rewards for defensive actions
            if o['ball_owned_player'] >= 0 and o['ball_owned_team'] == o['right_team']:
                # When an opponent player has the ball
                ball_pos = o['ball']
                player_pos = o['right_team'][o['ball_owned_player']]
                dist_to_ball = np.linalg.norm(ball_pos - player_pos)
                
                if dist_to_ball < 0.1:
                    # The agent is very close to the player with the ball
                    components["defensive_bonuses"][rew_index] += self.marking_bonus

                # Check for potential blocks if player is in a defensive role
                if o['right_team_roles'][o['ball_owned_player']] in self.defensive_positions:
                    # Assuming this would signal a block
                    if 'action' in o and o['action'] == 'slide':
                        components["defensive_bonuses"][rew_index] += self.blocking_bonus
            
            # Accumulate the rewards
            total_reward = sum([components[k][rew_index] for k in components])
            reward[rew_index] += total_reward

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
