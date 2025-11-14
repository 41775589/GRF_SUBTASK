from . import *
def build_scenario(builder):
    builder.config().game_duration = 400
    builder.config().deterministic = False
    builder.config().offsides = False
    builder.config().end_episode_on_score = True
    builder.config().end_episode_on_out_of_play = True
    builder.config().end_episode_on_possession_change = True

    builder.SetBallPosition(0.5, 0.0)  # Ball starts near the midfield

    # Set the team for the left (our team)
    builder.SetTeam(Team.e_Left)
    builder.AddPlayer(-1.0, 0.0, e_PlayerRole_GK)  # Goalkeeper
    builder.AddPlayer(0.0, 0.0, e_PlayerRole_CB, controllable=True)  # Controllable center back to train on Sliding and Stop-moving

    # Set the team for the right (opponent team)
    builder.SetTeam(Team.e_Right)
    builder.AddPlayer(-1.0, 0.0, e_PlayerRole_GK)  # Opponent goalkeeper
    # Adding multiple attackers from the opposition to simulate high-pressure situations
    builder.AddPlayer(0.5, -0.3, e_PlayerRole_CF)
    builder.AddPlayer(0.5, 0.3, e_PlayerRole_CF)
