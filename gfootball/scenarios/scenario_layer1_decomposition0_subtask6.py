from . import *
def build_scenario(builder):
    builder.config().game_duration = 400
    builder.config().deterministic = False  # Allow variability in player interactions
    builder.config().offsides = False  # Simplify rules for focused training on dribbling
    builder.config().end_episode_on_score = True
    builder.config().end_episode_on_out_of_play = True
    builder.config().end_episode_on_possession_change = False  # Continue the episode despite changes in possession to emphasize continuous control

    builder.SetBallPosition(0.1, 0.0)  # Start near the center but slightly to the right

    builder.SetTeam(Team.e_Left)
    builder.AddPlayer(-1.0, 0.0, e_PlayerRole_GK)  # Goalkeeper not focus, far from action
    builder.AddPlayer(0.1, 0.0, e_PlayerRole_CF, controllable=True)  # Sole agent in a dribbling focused role

    builder.SetTeam(Team.e_Right)
    builder.AddPlayer(-1.0, 0.0, e_PlayerRole_GK)  # Opponent goalkeeper
    # Add defensive players who apply pressure but are not overly aggressive
    builder.AddPlayer(0.0, 0.1, e_PlayerRole_CB, controllable=False)  
    builder.AddPlayer(0.0, -0.1, e_PlayerRole_CB, controllable=False)
    builder.AddPlayer(0.2, 0.2, e_PlayerRole_DM, controllable=False)  # Distant player to add slight positional challenge
