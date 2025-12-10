import time
import numpy as np
import argparse
import curses  # terminal key capture on Linux/Mac
from Info_relay_env_v2 import Info_relay_env

"""
Usage:
run ./info_relay.py to run the environment with random actions.
Just sit back and watch

run ./info_relay.py --keyboard and press any directional key to
bring up the gui

"""

parser = argparse.ArgumentParser()
parser.add_argument("-k", "--keyboard", action="store_true",
                    help="Enable keyboard control instead of random actions")
args = parser.parse_args()

USE_KEYBOARD = args.keyboard   # Default = False unless -k passed

def policy(agent, observations): 
    #return np.random.uniform(low=0, high=1, size=Info_relay_env.action_space(agent).shape)
    return np.ones(Info_relay_env.action_space(agent).shape)

def get_keyboard_action(stdscr, action_mapping):
    """
    Wait until a key is pressed, then return the corresponding action index.
    Arrow keys → move
    P/p → idle (0)
    Other keys → ignored (loop continues)
    """
    stdscr.nodelay(False)  # blocking mode: getch() waits until a key is pressed

    keyboard_map = {
        curses.KEY_RIGHT: (1, 0),
        curses.KEY_LEFT:  (2, 0),
        curses.KEY_UP:    (0, 1),
        curses.KEY_DOWN:  (0, 2),
        ord(" "):         (0, 0),
        # Diagonals via WASD:
        ord('w'): (1, 1),  # up-right
        ord('W'): (1, 1),
        ord('a'): (2, 1),  # up-left
        ord('A'): (2, 1),
        ord('s'): (2, 2),  # down-left
        ord('S'): (2, 2),
        ord('d'): (1, 2),  # down-right
        ord('D'): (1, 2),
    }

    while True:
        key = stdscr.getch()  # **blocks until a key is pressed**
        if key in keyboard_map:
            # P key → idle action
            if key == ord("p") or key == ord("P"):
                return 0

            # Arrow key → map to discrete action
            vx, vy = keyboard_map[key]
            for idx, comb in action_mapping.items():
                if comb[0] == vx and comb[1] == vy:
                    return idx


def run_env(stdscr=None):
    # init curses only if keyboard mode
    if USE_KEYBOARD and stdscr is None:
        stdscr = curses.initscr()
        curses.cbreak()
        stdscr.keypad(True)
        stdscr.nodelay(True)

    env = Info_relay_env(
        num_agents=1, num_emitters=0, num_bases=2, max_cycles=100,
        com_used=True, antenna_used=False, continuous_actions=False,
        num_messages=1, render_mode="human", using_half_velocity=True,
        step_size=1, a_max=0.1, num_CL_episodes = 0, num_r_help_episodes = 0
    )

    obs, info = env.reset()

    try:
        while env.agents:
            if USE_KEYBOARD:
                action = get_keyboard_action(stdscr, env.action_mapping_dict)
                actions = {"agent_0": action}
            else:
                actions = {a: env.action_space(a).sample() for a in env.agents}

            obs, reward, term, trunc, info = env.step(actions)

            #print(f"Reward: {reward['agent_0']:.4f} \n", flush = True)
            #print(f"Observation: {obs} \n", flush = True)
            with open ("out.txt", 'a') as f:
                f.write(f"Reward: {reward['agent_0']:.4f} \n")
                f.write(f"Observation: {obs} \n")
                f.flush()

            if term.get("agent_0"):
                print("\nEpisode ended.")
                break

            time.sleep(0.05)

    finally:
        env.close()

        if USE_KEYBOARD:
            stdscr.keypad(False)
            curses.nocbreak()
            curses.echo()
            curses.endwin()


# ENTRY
run_env()