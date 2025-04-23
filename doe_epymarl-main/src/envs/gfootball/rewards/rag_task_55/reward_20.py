import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that rewards successful tackles made by the agent, emphasizing avoidance of fouls."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)  # For tracking which sticky actions are being used
        self.tackle_successful_bonus = 0.5  # Bonus for successfully tackling without a foul
        self.foul_penalty = -0.5  # Penalty for committing a foul
        self.last_ball_possession_team = None  # Track which team had the ball last for determining tackles

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.last_ball_possession_team = None
        return self.env.reset()

    def get_state(self, to_pickle):
        state = self.env.get_state(to_pickle)
        state['last_ball_possession_team'] = self.last_ball_possession_team
        return state

    def set_state(self, state):
        self.last_ball_possession_team = state.get('last_ball_possession_team', None)
        return self.env.set_state(state)

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "tackle_bonus": 0.0,
                      "foul_penalty": 0.0}

        for i, obs in enumerate(observation):
            current_ball_possession = obs.get('ball_owned_team', -1)
            player_has_ball = obs.get('designated', -1) == obs.get('active', -1)
            game_mode = obs['game_mode']

            # Detect change of possession to our defending agent from the other team
            if current_ball_possession == 0 and self.last_ball_possession_team == 1 and player_has_ball:
                # Tackle encountered, check for foul
                if game_mode in [3, 6]:  # Game modes correspond to FreeKick or Penalty implying a foul
                    components["foul_penalty"] -= self.foul_penalty
                else:
                    components["tackle_bonus"] += self.tackle_successful_bonus
            
            # Update last ball possession state
            self.last_ball_possession_team = current_ball_possession

        reward += components["tackle_bonus"]
        reward += components["foul_penalty"]
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
                self.sticky_actions_counter[i] += action
        
        return observation, reward, done, info
