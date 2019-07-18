# http://vizdoom.cs.put.edu.pl/tutorial
# shortest working example
# [2019-07-17 18:40]

from vizdoom import DoomGame
import random
import time

game = DoomGame()
game.load_config("./basic.cfg")
game.init()

shoot   = [0,0,1]
left    = [1,0,0]
right   = [0,1,0]
actions = [shoot, left, right]

episodes = 10
for i in range(episodes):
    game.new_episode()
    while not game.is_episode_finished():
        s = game.get_state()
        img = s.game_variables
        r = game.make_action(random.choice(actions))
        print("-> Reward: %.3f" % r)
        time.sleep(0.02)
    print("GG: Result (Ret): %.3f" % game.get_total_reward())
    time.sleep(1)