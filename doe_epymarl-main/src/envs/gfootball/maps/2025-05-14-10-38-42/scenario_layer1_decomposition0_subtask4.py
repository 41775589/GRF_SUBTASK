from . import *
def build_scenario(builder):
    # Set configuration parameters
    builder.config().game_duration = 600  # Longer duration to practice various scenarios
    builder.config().deterministic = False  # Non-deterministic gameplay for robustness
    builder.config().offsides = False  # Removes offsides to focus on defensive plays
    builder.config().end_episode_on_score = True  # Ends the episode when a goal is scored
    builder.config().end_episode_on_out_of_play = True  # Ends the episode when the ball goes out
    builder.config().end_episode_on_possession_change = False  # Continue after possession change for continuous play

    # Ball setup towards the defensive line to initiate attack from opponents
    builder.SetBallPosition(-0.4, 0.0)

    # Defensive team setup (left side as per convention)
    builder.SetTeam(Team.e_Left)
    builder.AddPlayer(-1.0, 0.0, e_PlayerRole_GK, controllable=False)  # Goalkeeper
    builder.AddPlayer(-0.5, 0.1, e_PlayerRole_CB)  # Centre Back near the right on the field
    builder.AddPlayer(-0.5, -0.1, e_PlayerRole_CB)  # Centre Back near the left on the field

    # Offensive opponents (right side)
    builder.SetTeam(Team.e_Right)
    builder.AddPlayer(-1.0, 0.0, e_PlayerRole_GK, controllable=False)  # Opponent's goalkeeper
    builder.AddPlayer(-0.3, 0.15, e_PlayerRole_CF)  # Opponent's center forward right
    builder.AddPlayer(-0.3, -0.15, e_PlayerRole_CF)  # Opponent's center forward left
    builder.AddPlayer(-0.3, 0.0, e_PlayerRole_AM)  # Attacking midfielder to initiate plays
