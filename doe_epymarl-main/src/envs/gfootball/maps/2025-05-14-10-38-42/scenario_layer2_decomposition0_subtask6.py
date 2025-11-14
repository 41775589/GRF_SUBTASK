from . import *
def build_scenario(builder):
    builder.config().game_duration = 400
    builder.config().deterministic = False
    builder.config().offsides = False
    builder.config().end_episode_on_score = True
    builder.config().end_episode_on_out_of_play = True
    builder.config().end_episode_on_possession_change = True

    # Set the initial position of the ball near the goal to simulate close-range defense scenarios
    builder.SetBallPosition(-0.9, 0.0)

    # Configure the left team (controlled team)
    builder.SetTeam(Team.e_Left)
    builder.AddPlayer(-1.0, 0.0, e_PlayerRole_GK)  # Goalkeeper
    # Single defensive player focusing on close-range defensive skills like sliding
    builder.AddPlayer(-0.8, 0.0, e_PlayerRole_CB, controllable=True)

    # Configure the right team (opposing team)
    builder.SetTeam(Team.e_Right)
    builder.AddPlayer(-1.0, 0.0, e_PlayerRole_GK)  # Opposing goalkeeper 
    # Forward players positioned close to simulate pressure near the goal
    builder.AddPlayer(-0.75, 0.1, e_PlayerRole_CF, controllable=False)
    builder.AddPlayer(-0.75, -0.1, e_PlayerRole_CF, controllable=False)
