from . import *
def build_scenario(builder):
    # Configurations specific to this game scenario
    builder.config().game_duration = 400  # Duration of the game in simulation steps
    builder.config().deterministic = False  # Random elements in game dynamics
    builder.config().offsides = False  # No offsides rule to simplify movements
    builder.config().end_episode_on_score = True  # End the episode when a goal is scored
    builder.config().end_episode_on_out_of_play = True  # End the episode when the ball goes out of play
    builder.config().end_episode_on_possession_change = True  # End when possession changes
    
    # Setting ball position near the midfield to start offensive plays
    builder.SetBallPosition(-0.5, 0.0) 

    # Setting up the left team (our trained agents' team)
    builder.SetTeam(Team.e_Left)
    builder.AddPlayer(-1.0, 0.0, e_PlayerRole_GK)  # Adding goalkeeper
    builder.AddPlayer(-0.4, 0.2, e_PlayerRole_CF)  # Center Forward right position
    builder.AddPlayer(-0.4, -0.2, e_PlayerRole_CF)  # Center Forward left position

    # Setting up the right team (opponent team controlled by the computer)
    builder.SetTeam(Team.e_Right)
    builder.AddPlayer(-1.0, 0.0, e_PlayerRole_GK)  # Opponent goalkeeper
    builder.AddPlayer(-0.6, 0.1, e_PlayerRole_CB)  # Center Back, right side
    builder.AddPlayer(-0.6, -0.1, e_PlayerRole_CB)  # Center Back, left side
