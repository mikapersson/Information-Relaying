#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#
import argparse
from pathlib import Path

from benchmarl.hydra_config import reload_experiment_from_file

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluates the experiment from a checkpoint file."
    )
    parser.add_argument(
        "checkpoint_file", type=str, help="The name of the checkpoint file"
    )
    args = parser.parse_args()
    checkpoint_file = str(Path(args.checkpoint_file).resolve())
    experiment = reload_experiment_from_file(checkpoint_file)

    experiment.config.evaluation_episodes = 1000 # OBS used to evaluate over more episodes. Does not overwrite the old config files! :)
    experiment.logger.calculate_extra = True
    experiment.evaluate()
    
    # if experiment.task.has_render(experiment.test_env) and experiment.config.render:
    #             video_frames = []

    #             def callback(env, td):
    #                 video_frames.append(
    #                     experiment.task.__class__.render_callback(experiment, env, td)
    #                 )

    # else:
    #     video_frames = None
    #     callback = None

    # if experiment.test_env.batch_size == ():
    #             rollouts = []
    #             for eval_episode in range(experiment.config.evaluation_episodes):
    #                 rollouts.append(
    #                     experiment.test_env.rollout(
    #                         max_steps=experiment.max_steps,
    #                         policy=experiment.policy,
    #                         callback=callback if eval_episode == 0 else None,
    #                         auto_cast_to_device=True,
    #                         break_when_any_done=True,
    #                     )
    #                 )
    # else:
    #     rollouts = experiment.test_env.rollout(
    #         max_steps=experiment.max_steps,
    #         policy=experiment.policy,
    #         callback=callback,
    #         auto_cast_to_device=True,
    #         break_when_any_done=False,
    #         # We are running vectorized evaluation we do not want it to stop when just one env is done
    #     )

    # print(rollouts)

#outputs/2025-05-11/17-13-46/mappo_info_relay_mlp__71097d9f_25_05_11-17_13_46/checkpoints/checkpoint_1200000.pt
#outputs/2025-05-11/17-56-23/mappo_info_relay_mlp__694db0b9_25_05_11-17_56_23/checkpoints/checkpoint_3000000.pt
#outputs/2025-05-11/19-46-22/mappo_info_relay_mlp__0991dc4a_25_05_11-19_46_23/checkpoints/checkpoint_3000000.pt
#outputs/2025-05-11/21-58-16/mappo_info_relay_mlp__cfba64a3_25_05_11-21_58_16/checkpoints/checkpoint_9000000.pt
#outputs/2025-05-14/12-23-44/mappo_info_relay_mlp__ef637ad7_25_05_14-12_23_44/checkpoints/checkpoint_9000000.pt
#outputs/2025-05-06/12-27-29/mappo_info_relay_mlp__f73a0671_25_05_06-12_27_29/checkpoints/checkpoint_9000000.pt
#outputs/2025-05-08/22-31-06/mappo_info_relay_mlp__3e4d95ae_25_05_08-22_31_06/checkpoints/checkpoint_12000000.pt
#outputs/2025-05-09/22-02-07/mappo_info_relay_mlp__27520d5a_25_05_09-22_02_07/checkpoints/checkpoint_18000000.pt
#outputs/2025-05-07/13-14-19/mappo_info_relay_mlp__224c1d29_25_05_07-13_14_19/checkpoints/checkpoint_9000000.pt
#outputs/2025-05-08/07-36-37/mappo_info_relay_mlp__9542cabb_25_05_08-07_36_37/checkpoints/checkpoint_15000000.pt
#outputs/2025-05-09/07-36-53/mappo_info_relay_mlp__c029c709_25_05_09-07_36_53/checkpoints/checkpoint_18000000.pt
#outputs/2025-05-12/07-59-43/mappo_info_relay_mlp__ed0dbc8c_25_05_12-07_59_43/checkpoints/checkpoint_9000000.pt
#outputs/2025-05-12/14-29-15/mappo_info_relay_mlp__8aaddf00_25_05_12-14_29_15/checkpoints/checkpoint_15000000.pt
#outputs/2025-05-14/19-15-29/mappo_info_relay_mlp__84e99de6_25_05_14-19_15_29/checkpoints/checkpoint_15000000.pt
#outputs/2025-05-15/08-30-29/mappo_info_relay_mlp__69c75de4_25_05_15-08_30_29/checkpoints/checkpoint_12900000.pt

#outputs/2025-05-15/21-21-19/mappo_info_relay_mlp__2e2f4b04_25_05_15-21_21_19/checkpoints/checkpoint_12900000.pt
#outputs/2025-05-18/01-42-19/mappo_info_relay_mlp__e24c3ecd_25_05_18-01_42_19/checkpoints/checkpoint_12900000.pt
#outputs/2025-05-21/12-41-15/mappo_info_relay_mlp__35233c29_25_05_21-12_41_15/checkpoints/checkpoint_3000000.pt
#outputs/2025-05-21/20-30-19/mappo_info_relay_mlp__ff10acea_25_05_21-20_30_19/checkpoints/checkpoint_12000000.pt
#outputs/2025-05-22/13-48-16/mappo_info_relay_mlp__0675570a_25_05_22-13_48_16/checkpoints/checkpoint_12000000.pt 
#outputs/2025-05-24/14-26-57/mappo_info_relay_mlp__3c12ac04_25_05_24-14_26_57/checkpoints/checkpoint_18000000.pt
#outputs/2025-05-25/17-06-40/mappo_info_relay_mlp__caae4a32_25_05_25-17_06_40/checkpoints/checkpoint_12000000.pt
#outputs/2025-05-26/07-42-43/mappo_info_relay_mlp__0b394df5_25_05_26-07_42_44/checkpoints/checkpoint_6000000.pt
#outputs/2025-05-26/11-16-15/mappo_info_relay_mlp__093a885f_25_05_26-11_16_16/checkpoints/checkpoint_9000000.pt
#outputs/2025-05-26/19-57-51/mappo_info_relay_mlp__e4badb26_25_05_26-19_57_51/checkpoints/checkpoint_15000000.pt"