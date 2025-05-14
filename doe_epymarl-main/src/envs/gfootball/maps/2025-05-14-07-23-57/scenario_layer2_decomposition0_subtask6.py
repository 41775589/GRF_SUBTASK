from . import *
def build_scenario(builder):
    builder.config().game_duration = 400
    builder.config().deterministic = False
    builder.config().offsides = False
    builder.config().end_episode_on_score = True
    builder.config().end_episode_on_out_of_play = True
    builder.config().end_episode_on_possession_change = True

    # Setting the ball near the midfield to allow developmental focus on realistic game play
    builder.SetBallPosition(-0.4, 0.0)

    # Setting up the left team (our team - where the agent is playing)
    builder.SetTeam(Team.e_Left)
    builder.AddPlayer(-1.0, 0.0, e_PlayerRole_GK)  # Add a goalkeeper to comply with the rules
    builder.AddPlayer(-0.5, 0.0, e_PlayerRole_CB, controllable=True)  # Single agent focusing on sliding tactics

    # Setting up the right team (opponent team)
    builder.SetTeam(Team.e_Right)
    builder.AddPlayer(-1.0, 0.0, e_PlayerRole_GK)  # Opponent goalkeeper
    # Adding multiple opponents nearby to create frequent physical confrontations
    builder.AddPlayer(-0.4, 0.1, e_PlayerRole_CF)
    builder.AddPlayer(-0.4, -0.1, e_PlayerRole_CF)
    builder.AddPlayer(-0.3, 0.15, e_PlayerRole_CM)
    builder.AddPlayer(-0.3, -0.15, e_PlayerRole_CM)
