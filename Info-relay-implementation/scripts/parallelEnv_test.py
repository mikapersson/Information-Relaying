from Info_relay_env_v2 import Info_relay_env

from pettingzoo.test import parallel_api_test

# https://pettingzoo.farama.org/content/environment_tests/
if __name__ == "__main__":
    env = Info_relay_env()
    parallel_api_test(env, num_cycles=1_000_000)