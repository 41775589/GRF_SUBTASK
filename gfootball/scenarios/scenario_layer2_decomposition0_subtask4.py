from . import *
def build_scenario(builder):
    builder.config().game_duration = 400
    builder.config().deterministic = False
    builder.config().offsides = False
    builder.config().end_episode_on_score = True
    builder.config().end_episode_on_out_of_play = True
    builder.config().end_episode_on_possession_change = True

    # Set up a simple scenario where the trained agent will practice slides in a one-on-one scenario
    builder.SetBallPosition(0.4, 0.0)  # Set the ball near the middle, but slightly towards the opponent's goal.

    builder.SetTeam(Team.e_Left)
    builder.AddPlayer(-1.0, 0.0, e_PlayerRole_GK)  # Adding goalkeeper to the left team.
    builder.AddPlayer(0.3, 0.0, e_PlayerRole_CB)   # Our agent, a center back, positioned to initiate a sliding tackle.

    builder.SetTeam(Team.e_Right)
    builder.AddPlayer(-1.0, 0.0, e_PlayerRole_GK)  # Adding goalkeeper to the right team as well.
    builder.AddPlayer(0.5, 0.0, e_PlayerRole_CF)   # Add an opposing player who will receive the ball and be targeted for sliding.
