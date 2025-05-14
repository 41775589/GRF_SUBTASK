from . import *
def build_scenario(builder):
    builder.config().game_duration = 400
    builder.config().deterministic = False
    builder.config().offsides = False
    builder.config().end_episode_on_score = True
    builder.config().end_episode_on_out_of_play = True
    builder.config().end_episode_on_possession_change = False

    # Initialize the ball position near the opponent to encourage defensive play
    builder.SetBallPosition(0.5, 0.0)

    # Set up the left team (our team)
    builder.SetTeam(Team.e_Left)
    builder.AddPlayer(-1.0, 0.0, e_PlayerRole_GK)  # our goalkeeper
    builder.AddPlayer(0.3, 0.0, e_PlayerRole_DM, controllable=True)  # player supposed to practice "Sliding"

    # Set up the right team (opponent)
    builder.SetTeam(Team.e_Right)
    builder.AddPlayer(-1.0, 0.0, e_PlayerRole_GK)  # opponent goalkeeper
    builder.AddPlayer(0.6, 0.0, e_PlayerRole_CF)  # Opponent forward
