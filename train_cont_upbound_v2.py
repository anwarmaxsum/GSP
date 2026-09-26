"""Train an algorithm."""
import argparse
import json
from harl.utils.configs_tools import get_defaults_yaml_args, update_args

import copy

import os
import random

import time

def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--algo",
        type=str,
        default="happo",
        choices=[
            "happo",
            "hatrpo",
            "haa2c",
            "haddpg",
            "hatd3",
            "hasac",
            "had3qn",
            "maddpg",
            "matd3",
            "mappo",

            "macpro",


            "happo_ewc",
            "happo_si",
            "happo_mas",

            "happo_ewc_mh",
        ],
        help="Algorithm name. Choose from: happo, hatrpo, haa2c, haddpg, hatd3, hasac, had3qn, maddpg, matd3, mappo.",
    )
    parser.add_argument(
        "--env",
        type=str,
        default="pettingzoo_mpe",
        choices=[
            "smac",
            "mamujoco",
            "pettingzoo_mpe",
            "gym",
            "football",
            "dexhands",
            "smacv2",
            "lag",
            "mpe2",
            "atari",

            "bsk",
            "bsk_single_sat",
            "bsk_scarlet",
        ],
        help="Environment name. Choose from: smac, mamujoco, pettingzoo_mpe, gym, football, dexhands, smacv2, lag.",
    )
    parser.add_argument(
        "--exp_name", type=str, default="installtest", help="Experiment name."
    )
    parser.add_argument(
        "--load_config",
        type=str,
        default="",
        help="If set, load existing experiment config file instead of reading from yaml config file.",
    )

    parser.add_argument('--sz', action='store_true')
    parser.add_argument('--sz1', action='store_true')
    parser.add_argument('--reusenet', action='store_true')

    args, unparsed_args = parser.parse_known_args()


    def process(arg):
        try:
            return eval(arg)
        except:
            return arg

    keys = [k[2:] for k in unparsed_args[0::2]]  # remove -- from argument
    values = [process(v) for v in unparsed_args[1::2]]
    unparsed_dict = {k: v for k, v in zip(keys, values)}
    args = vars(args)  # convert to dict
    if args["load_config"] != "":  # load config from existing config file
        with open(args["load_config"], encoding="utf-8") as file:
            all_config = json.load(file)
        args["algo"] = all_config["main_args"]["algo"]
        args["env"] = all_config["main_args"]["env"]
        algo_args = all_config["algo_args"]
        env_args = all_config["env_args"]
    else:  # load config from corresponding yaml file
        algo_args, env_args = get_defaults_yaml_args(args["algo"], args["env"])
    update_args(unparsed_dict, algo_args, env_args)  # update args from command line

    if args["env"] == "dexhands":
        import isaacgym  # isaacgym has to be imported before PyTorch

    # note: isaac gym does not support multiple instances, thus cannot eval separately
    if args["env"] == "dexhands":
        algo_args["eval"]["use_eval"] = False
        algo_args["train"]["episode_length"] = env_args["hands_episode_length"]

    # start training
    from harl.runners import RUNNER_REGISTRY

    if args["env"] == "mamujoco":
        task_var1 = "scenario"
        task_var2 = "agent_conf"
        # task_var1_list = ["Ant-v2","Walker2d-v2", "HalfCheetah-v2","Hopper-v2"]
        # task_var2_list = ["4x2","2x3","2x3", "3x1"]

        # task_var1_list = ["Hopper-v2","Walker2d-v2", "HalfCheetah-v2", "Ant-v2"]
        # task_var2_list = ["3x1","2x3","2x3", "2x4"]

        #task_var1_list = ["Swimmer-v2", "Walker2d-v2", "HalfCheetah-v2", "Ant-v2"]
        #task_var2_list = ["2x1", "2x3","2x3", "2x4"]
        #algo_args["train"]["num_env_steps"] = 1000000


        #R3
        # task_var1_list = ["Swimmer-v2", "Walker2d-v2", "HalfCheetah-v2", "Ant-v2"]
        # task_var2_list = ["2x1", "2x3","2x3", "2x4"]

        # R5
        # task_var1_list = ["Swimmer-v2", "HalfCheetah-v2", "Ant-v2", "Hopper-v2"]
        # task_var2_list = ["2x1", "2x3", "2x4", "3x1"]

        #R6
        # task_var1_list = ["Swimmer-v2", "HalfCheetah-v2", "Ant-v2", "HalfCheetah-v2"]
        # task_var2_list = ["2x1", "2x3", "2x4", "3x2"]

        #R7
        task_var1_list = ["Swimmer-v2", "HalfCheetah-v2", "Ant-v2", "Ant-v2"]
        task_var2_list = ["2x1", "2x3", "2x4", "4x2"]
        # algo_args["train"]["num_env_steps"] = 1000000
        algo_args["model"]["hidden_sizes"] = [160,160,176]


    elif args["env"] == "football":
        task_var1 = "env_name"
        task_var2 = "number_of_left_players_agent_controls"
        task_var1_list = ["academy_pass_and_shoot_with_keeper", "academy_run_pass_and_shoot_with_keeper", "academy_3_vs_1_with_keeper", "academy_counterattack_easy"]
        # task_var1_list = ["academy_pass_and_shoot_with_keeper", "academy_3_vs_1_with_keeper", "academy_counterattack_easy", "academy_counterattack_hard"]
        # task_var1_list = ["academy_run_pass_and_shoot_with_keeper", "academy_3_vs_1_with_keeper", "academy_counterattack_easy","academy_pass_and_shoot_with_keeper"]
        # task_var1_list = ["academy_pass_and_shoot_with_keeper", "academy_3_vs_1_with_keeper", "academy_counterattack_easy", "academy_counterattack_hard"]
        task_var2_list = [2,2,3,4]
        algo_args["train"]["num_env_steps"] = 5000000
        algo_args["train"]["n_rollout_threads"] = 50
        algo_args["train"]["episode_length"] = 200
        algo_args["train"]["log_interval"] = 10
        algo_args["train"]["eval_interval"] = 10

        algo_args["model"]["hidden_sizes"] = [64,64]
        # algo_args["model"]["use_recurrent_policy"] = False

        algo_args["algo"]["actor_num_mini_batch"] = 2
        algo_args["algo"]["critic_num_mini_batch"] = 2
        algo_args["algo"]["ppo_epoch"] = 15
        algo_args["algo"]["critic_epoch"] = 15


        # task_var2_list = [2,3,4,4]

    elif args["env"] == "smac":
        task_var1 = "map_name"
        # task_var1_list = ["3m","5m_vs_6m","2s3z","6h_vs_8z"] #not working ["3s5z_vs_3s6z","8m_vs_9m", "10m_vs_11m"]
        # task_var1_list = ["3m","5m_vs_6m","8m","8m_vs_9m"]
        # task_var1_list = ["3s_vs_3z","3s_vs_4z","3s5z","3s5z_vs_3s6z"] 
        # task_var1_list = ["2s_vs_1sc","2s3z","2s4z","1c3s5z"] 
        if args["sz"] :
            task_var1_list = ["1s2z","2s1z_vs_3z","2s3z","2s4z"]
        elif args["sz1"] :
            # task_var1_list = ["1s2z","2s3z","2s4z","3s5z"]
            task_var1_list = ["1s2z","2s2z","2s3z","2s4z"]
        else:
            task_var1_list = ["3m","5m","7m","8m"]
        

    elif args["env"] == "pettingzoo_mpe":
        task_var1 = "scenario"
        # task_var1_list = ["simple_v2", "simple_spread_v2", "simple_reference_v2", "simple_speaker_listener_v3"]
        # task_var1_list = ["simple_v2", "simple_spread_v2", "simple_speaker_listener_v3", "simple_reference_v2"]
        # task_var1_list = ["simple_reference_v2", "simple_spread_v2", "simple_speaker_listener_v3"]
        # task_var1_list = ["simple_v2", "simple_speaker_listener_v3", "simple_reference_v2", "simple_spread_v2"]
        # task_var1_list = ["simple_v2", "simple_speaker_listener_v3",  "simple_spread_v2", "simple_reference_v2"]
        task_var1_list = ["simple_reference_v2", "simple_speaker_listener_v3", "simple_spread_v2", "simple_spread_v2"]

    elif args["env"] == "mpe2":
        task_var1 = "scenario"
        # task_var1_list = ["simple_reference_v3", "simple_speaker_listener_v4", "simple_spread_v3", "simple_formation_v1"]

        # task_var1_list = ["simple_reference_v3", "simple_spread_v3", "simple_line_v1", "simple_formation_v1"]
        task_var1_list = ["simple_reference_v3", "simple_speaker_listener_v4", "simple_line_v1", "simple_formation_v1"]

    elif args["env"] == "atari":
        task_var1 = "scenario"
        task_var1_list = ["entombed_cooperative_v3", "joust_v3", "mario_bros_v3", "wizard_of_wor_v3"]
        # task_var1_list = ["mario_bros_v3", "wizard_of_wor_v3"]
        # task_var1_list = ["joust_v3", "mario_bros_v3", "wizard_of_wor_v3"]
        algo_args["model"]["max_obs_length"] = 100800 #max_act_length: 20
        algo_args["model"]["max_act_length"] = 20

    elif args["env"] == "bsk":
        task_var1 = "key"
        # task_var1_list = ["het_cluster-hard-random_res", "het_cluster-hard-random_res"]
        # task_var1_list = ["het_cluster-easy","het_cluster-easy-random_res","het_cluster-hard", "het_cluster-hard-random_res"]
        # task_var1_list = ["het_cluster-easy","het_cluster-medium","het_cluster-hard", "het_cluster-hard-random_res"] #
        task_var1_list = ["het_cluster-easy","het_cluster-medium","het_cluster-hard", "het_cluster-xhard"] #
        algo_args["train"]["n_rollout_threads"] = 40
        algo_args["train"]["num_env_steps"] = 300000
        algo_args["eval"]["n_rollout_threads"] = 10
        algo_args["eval"]["eval_episodes"] = 5

    train_runner = []

    start_time = time.time()
    print("algo_args: ",algo_args)

    for i in range(0,len(task_var1_list)):
        
        env_args[task_var1] = task_var1_list[i]
        # if i == 3 and task_var1_list[i] == "simple_speaker_listener_v3":
        #     print("simple_speaker_listener_v3 with discrete ")
        #     env_args["continuous_actions"] = False

        if args["env"] == "mamujoco" or args["env"] == "football":
            env_args[task_var2] = task_var2_list[i]

        if args["env"] == "pettingzoo_mpe" and i==3:
            # env_args["N"] = 5
            env_args["N"] = 4
        elif "N" in env_args.keys():
            del env_args["N"]
        
        if args["env"] == "mamujoco" and i==1:
            algo_args["seed"]["ori"] = algo_args["seed"]["seed"]
            algo_args["seed"]["seed"] = 3
        elif args["env"] == "mamujoco" and i==2:
            algo_args["seed"]["seed"] = algo_args["seed"]["ori"]

        runner = RUNNER_REGISTRY[args["algo"]](args, algo_args, env_args)
        
        train_runner.append(runner)
        print("runner actor: \n", runner.actor, "------------")
        
        if i > 0 and args["reusenet"]:
            model_transfer(train_runner[i-1], train_runner[i],args)  

        # if i > 0:
        #     model_transfer(train_runner[i-1], train_runner[i],args)   
        #     train_runner[i].logger = train_runner[i-1].logger   
        #     train_runner[i].save_dir = train_runner[i-1].save_dir 
        # print("save dir task i : ", i,train_runner[i].save_dir)


        runner.run()
        # print("Current Runner eval after run")
        # runner.eval()

        for j in range(0,i+1):
            # if i > 0:
            #     # pass
            #     model_transfer(train_runner[i], train_runner[j],args,False) 
            print("Eval task: ",j, " on task: ", i,end=" ")
            eval_runner = train_runner[j]
            eval_runner.eval()

        # if args["env"] == "smac":
        #     os.system("pkill -f SC2_x64")

        print("save dir task i : ", i,train_runner[i].save_dir)

    for i in range(0, len(train_runner)):
        # print("save dir task i : ", i,train_runner[i].save_dir)
        train_runner[i].close()
        # runner.close()

    print("--- Total running time: %s seconds ---" % (time.time() - start_time))

    # runner = RUNNER_REGISTRY[args["algo"]](args, algo_args, env_args)
    # runner.run()
    # runner.close()


def model_transfer(src_runner, dst_runner, args, isTraining=True):
    # print("call model transfer, isTraining = ",isTraining, " algo = ", args["algo"])
    choices=[
            "happo",
            "hatrpo",
            "haa2c",
            "haddpg",
            "hatd3",
            "hasac",
            "had3qn",
            "maddpg",
            "matd3",
            "mappo",

            "macpro",

            "happo_ewc",
            "happo_si",
            "happo_mas",

            "happo_ewc_mh",
        ],

    # print("check if algo in choices: ", any(args["algo"] in x  for x in choices))
    if isTraining:
        # if args["algo"] == "happo":
        if any(args["algo"] in x  for x in choices):
        # if args["algo"] == "happo":
            print("Model weight transfer for training, algo = ", args["algo"])
            for i in range (0,len(dst_runner.actor)):
                # if i < len(src_runner.actor):
                #     dst_runner.actor[i].actor.base.load_state_dict(src_runner.actor[i].actor.base.state_dict())
                #     if hasattr(dst_runner.actor[i].actor, 'rnn'):
                #        dst_runner.actor[i].actor.rnn.load_state_dict(src_runner.actor[i].actor.rnn.state_dict()) 
                # else:
                #     idx = random.randint(0,src_runner.num_agents-1)
                #     dst_runner.actor[i].actor.base.load_state_dict(src_runner.actor[idx].actor.base.state_dict())
                #     if hasattr(dst_runner.actor[i].actor, 'rnn'):
                #         dst_runner.actor[i].actor.rnn.load_state_dict(src_runner.actor[idx].actor.rnn.state_dict())

                if i < len(src_runner.actor):
                    dst_runner.actor[i].actor.load_state_dict(src_runner.actor[i].actor.state_dict())
                else:
                    idx = random.randint(0,src_runner.num_agents-1)
                    dst_runner.actor[i].actor.load_state_dict(src_runner.actor[idx].actor.state_dict())

            dst_runner.critic.critic.load_state_dict(src_runner.critic.critic.state_dict())
    else:
        if any(args["algo"] in x  for x in choices):
        # if args["algo"] == "happo":
            print("Model weight transfer for evaluation, algo = ", args["algo"])
            for i in range (0,min(dst_runner.num_agents,src_runner.num_agents)):
                    #dst_runner.actor[i].actor.base.load_state_dict(src_runner.actor[i].actor.base.state_dict())
                    dst_runner.actor[i].actor.load_state_dict(src_runner.actor[i].actor.state_dict())

            dst_runner.critic.critic.load_state_dict(src_runner.critic.critic.state_dict())
            

if __name__ == "__main__":
    main()
