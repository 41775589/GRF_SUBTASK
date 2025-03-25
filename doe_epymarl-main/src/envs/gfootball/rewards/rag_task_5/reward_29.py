import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper to emphasize defensive skills and counter-attacking in football agents."""

    def __init__(self, env):
        super().__init__(env)
        self.num_zones = 10  # Number of defensive and counter-attack zones
        self.defensive_reward = 0.1
        self.transition_reward = 0.15
        self.zone_thresholds = np.linspace(-1, 1, self.num_zones+1)
        self.collected_zones = {}
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        # Reset the reward zones collection state and sticky actions.
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.collected_zones = {}
        return self.env.reset()

    def get_state(self, to_pickle):
        # Save the current state of collected reward zones.
        to_pickle['collected_zones'] = self.collected_zones
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        # Restore the state of collected reward zones.
        state = self.env.set_state(state)
        self.collected_zones = state.get('collected_zones', {})
        return state

    def reward(self, reward):
        # Calculate additional rewards based on defensive and counter-attacking play.
        obs = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "defensive_reward": [0.0] * len(reward),
                      "transition_reward": [0.0] * len(reward)}

        if obs is None:
            return reward, components

        assert len(reward) == len(obs)

        for agent_index in range(len(reward)):
            player_obs = obs[agent_index]
            ball_pos = player_obs['ball'][0]  # x-axis position of the ball
            
            # Collect defensive rewards based on ball position in own half.
            if ball_pos < 0:
                zone_index = np.digitize(ball_pos, self.zone_thresholds) - 1
                if zone_index not in self.collected_zones.get(agent_index, []):
                    if player_obs['ball_owned_team'] in [0, 1]:  # Ball possession by either team
                        components["defensive_reward"][agent_index] = self.defensive_reward
                        reward[agent_index] += components["defensive_reward"][agent_index]
                        # Mark this zone as collected for this episode
                        self.collected_zones.setdefault(agent_index, []).append(zone_index)

            # Collect transition rewards for initiating counter-attacks.
            if player_obs['ball_owned_team'] == 0 and ball_pos >= 0.5:  # Team 0 starting a counter from their half
                transition_zone = np.digitize(ball_pos, self.zone_thresholds)
                if transition_zone not in self.collected_zones.get(agent_index, []):
                    components["transition_reward"][agent_index] = self.transition_reward
                    reward[agent_index] += components["transition_reward"][agent_index]
                    self.collected_zones.setdefault(agent_index, []).append(transition_zone)

        return reward, components

    def step(self, action):
        # Process a step of the environment, adding detailed reward breakdown into info.
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action_active in enumerate(agent_obs['sticky_actions']):
                self.sticky_actions_counter[i] += action_active
                info[f"sticky_actions_{i}"] = self.sticky_actions_counter[i]
        return observation, reward, done, info
