import datetime
import os
from os.path import dirname, abspath
import pprint
import shutil
import time
import threading
from types import SimpleNamespace as SN

import torch as th
import copy

from controllers import REGISTRY as mac_REGISTRY
from components.episode_buffer import ReplayBuffer
from components.transforms import OneHot
from learners import REGISTRY as le_REGISTRY
from runners import REGISTRY as r_REGISTRY
from utils.general_reward_support import test_alg_config_supports_reward
from utils.logging import Logger
from utils.timehelper import time_left, time_str

from modules.doe import doe_classifier_config_loader


def run(_run, _config, _log):
    # check args sanity
    _config = args_sanity_check(_config, _log)

    args = SN(**_config)
    args.device = "cuda" if args.use_cuda else "cpu"
    print("AAAAAAAAAA",args)
    assert test_alg_config_supports_reward(
        args
    ), "The specified algorithm does not support the general reward setup. Please choose a different algorithm or set `common_reward=True`."

    # setup loggers
    logger = Logger(_log)

    _log.info("Experiment Parameters:")
    experiment_params = pprint.pformat(_config, indent=4, width=1)
    _log.info("\n\n" + experiment_params + "\n")

    # configure tensorboard logger
    # unique_token = "{}__{}".format(args.name, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))

    try:
        map_name = _config["env_args"]["map_name"]
    except:
        map_name = _config["env_args"]["key"]
    unique_token = (
        f"{_config['name']}_seed{_config['seed']}_{map_name}_{datetime.datetime.now()}"
    )

    args.unique_token = unique_token
    if args.use_tensorboard:
        tb_logs_direc = os.path.join(
            dirname(dirname(abspath(__file__))), "results", "tb_logs",args.time_stamp
        )
        tb_exp_direc = os.path.join(tb_logs_direc, f'layer{args.layer_id}_decomposition{args.decomposition_id}_subtask{args.group_id}_iter{args.iter_id}_sample{args.sample_id}')
        logger.setup_tb(tb_exp_direc)

    if args.use_wandb:
        logger.setup_wandb(
            _config, args.wandb_team, args.wandb_project, args.wandb_mode
        )

    # sacred is on by default
    logger.setup_sacred(_run)

    # Run and train

    run_sequential(args=args, logger=logger)

    # Clean up after finishing
    print("Exiting Main")

    print("Stopping all threads")
    for t in threading.enumerate():
        if t.name != "MainThread":
            print("Thread {} is alive! Is daemon: {}".format(t.name, t.daemon))
            t.join(timeout=1)
            print("Thread joined")

    print("Exiting script")

    # Making sure framework really exits
    # os._exit(os.EX_OK)


def evaluate_sequential(args, runner):
    for _ in range(args.test_nepisode):
        runner.run(test_mode=True)

    if args.save_replay:
        runner.save_replay()

    runner.close_env()


def run_sequential(args, logger):
    # Init runner so we can get env info
    runner = r_REGISTRY[args.runner](args=args, logger=logger)

    # Set up schemes and groups here
    env_info = runner.get_env_info()
    args.n_agents = env_info["n_agents"]
    args.n_actions = env_info["n_actions"]
    args.state_shape = env_info["state_shape"]

    # Default/Base scheme
    scheme = {
        "state": {"vshape": env_info["state_shape"]},
        "obs": {"vshape": env_info["obs_shape"], "group": "agents"},
        "actions": {"vshape": (1,), "group": "agents", "dtype": th.long},
        "avail_actions": {
            "vshape": (env_info["n_actions"],),
            "group": "agents",
            "dtype": th.int,
        },
        "terminated": {"vshape": (1,), "dtype": th.uint8},
    }
    # For individual rewards in gymmai reward is of shape (1, n_agents)
    if args.common_reward:
        scheme["reward"] = {"vshape": (1,)}
    else:
        scheme["reward"] = {"vshape": (args.n_agents,)}
    groups = {"agents": args.n_agents}
    preprocess = {"actions": ("actions_onehot", [OneHot(out_dim=args.n_actions)])}

    buffer = ReplayBuffer(
        scheme,
        groups,
        args.buffer_size,
        env_info["episode_limit"] + 1,
        preprocess=preprocess,
        device="cpu" if args.buffer_cpu_only else args.device,
    )

    # Setup multiagent controller here
    mac = mac_REGISTRY[args.mac](buffer.scheme, groups, args)

    # Give runner the scheme
    runner.setup(scheme=scheme, groups=groups, preprocess=preprocess, mac=mac)

    # Learner
    learner = le_REGISTRY[args.learner](mac, buffer.scheme, logger, args)


    # 如果要使用doe，那么加载对应agent的doe cls，并add到mac、learner
    # layer_tmp_dir 就是本层的所有pt文件根目录
    if hasattr(args, 'doe_classifier_cfg'):
        layer_tmp_dir = args.doe_classifier_cfg["layer_tmp_dir"]
        os.makedirs(layer_tmp_dir, exist_ok=True)
        # GRF_SUBTASK/doe_epymarl-main/results/gfootball/Time


    if args.use_doe:
        # 如果使用doe训练，那么在这里load doe cls；
        # 如果是第一次分解，显然不会用doe，也不用load doe cls
        # 如果是后续分解，不使用doe训练（采用普通merge训练的方法作为baseline），也不需要 load

        """
        这里考虑实现 merge multi doe classifier, 
        merge 操作放在 generator_one.py中，把那个cls存到这个path中
        并且名字要作为参数传入，或者作为args中传入
        这样可以保证run的逻辑不变，只需要 load 即可，不需要train和merge
        """

        # 假设已经merge完了，load doe cls "buffer_path/doe_name.pt"
        doe_classifier = doe_classifier_config_loader(
            n_agents=args.n_agents,
            cfg=args.doe_classifier_cfg,  # 本来是args.get("doe_classifier_cfg")，这里args是namespace形式
            buffer_path = layer_tmp_dir, # merge buffer load path,
            load_mode='load'
        )

        # 为 MAC 设置 DoE classifier
        if hasattr(mac, 'set_doe_classifier'):
            mac.set_doe_classifier(doe_classifier)

        # 为 Learner 设置 DoE classifier
        if hasattr(learner, 'set_doe_classifier'):
            learner.set_doe_classifier(doe_classifier)
        print("DoE_classifier is set to mac and learner")


    if args.use_cuda:
        learner.cuda()
    # 目前这个ckpt path为空，导致learner没有load model
    if args.checkpoint_path != "":
        # timesteps = []
        # timestep_to_load = 0
        #
        # if not os.path.isdir(args.checkpoint_path):
        #     logger.console_logger.info(
        #         "Checkpoint directiory {} doesn't exist".format(args.checkpoint_path)
        #     )
        #     return
        #
        # # Go through all files in args.checkpoint_path
        # for name in os.listdir(args.checkpoint_path):
        #     full_name = os.path.join(args.checkpoint_path, name)
        #     # Check if they are dirs the names of which are numbers
        #     if os.path.isdir(full_name) and name.isdigit():
        #         timesteps.append(int(name))
        #
        # if args.load_step == 0:
        #     # choose the max timestep
        #     timestep_to_load = max(timesteps)
        # else:
        #     # choose the timestep closest to load_step
        #     timestep_to_load = min(timesteps, key=lambda x: abs(x - args.load_step))
        # # !!!! 这个地方要改成加载的ckpt的路径和文件夹
        # model_path = os.path.join(args.checkpoint_path, str(timestep_to_load))
        #
        # logger.console_logger.info("Loading model from {}".format(model_path))
        # learner.load_models(model_path)
        # runner.t_env = timestep_to_load

        model_path = args.checkpoint_path
        logger.console_logger.info("Loading model from {}".format(model_path))
        learner.load_models(model_path)


        if args.evaluate or args.save_replay:
            runner.log_train_stats_t = runner.t_env
            evaluate_sequential(args, runner)
            logger.log_stat("episode", runner.t_env, runner.t_env)
            logger.print_recent_stats()
            logger.console_logger.info("Finished Evaluation")
            return

    # start training
    episode = 0
    last_test_T = -args.test_interval - 1
    last_log_T = 0
    model_save_time = 0

    start_time = time.time()
    last_time = start_time

    logger.console_logger.info("Beginning training for {} timesteps".format(args.t_max))

    # # 用于调试
    # args.t_max = 2000

    while runner.t_env <= args.t_max:
        # Run for a whole episode at a time
        episode_batch = runner.run(test_mode=False)
        buffer.insert_episode_batch(episode_batch)

        if buffer.can_sample(args.batch_size):
            episode_sample = buffer.sample(args.batch_size)

            # Truncate batch to only filled timesteps
            max_ep_t = episode_sample.max_t_filled()
            episode_sample = episode_sample[:, :max_ep_t]

            if episode_sample.device != args.device:
                episode_sample.to(args.device)

            learner.train(episode_sample, runner.t_env, episode)

        # Execute test runs once in a while
        n_test_runs = max(1, args.test_nepisode // runner.batch_size)
        if (runner.t_env - last_test_T) / args.test_interval >= 1.0:
            logger.console_logger.info(
                "t_env: {} / {}".format(runner.t_env, args.t_max)
            )
            logger.console_logger.info(
                "Estimated time left: {}. Time passed: {}".format(
                    time_left(last_time, last_test_T, runner.t_env, args.t_max),
                    time_str(time.time() - start_time),
                )
            )
            last_time = time.time()

            last_test_T = runner.t_env
            for _ in range(n_test_runs):
                runner.run(test_mode=True)

        if args.save_model and (
            runner.t_env - model_save_time >= args.save_model_interval
            or model_save_time == 0
        ):
            model_save_time = runner.t_env
            save_path = os.path.join(dirname(dirname(abspath(__file__))),
                                     args.local_results_path, "models",
                                     args.unique_token,
                                     str(runner.t_env))
            # 这个目录目前有点奇怪 '/data/qiaodan/projects/GRF_SUBTASK/doe_epymarl-main/results/models/ia2c_seed114514_scenario_layer2_decomposition0_subtask6_2025-05-05 00:15:46.173157/150'
            # 需要改成 GRF_SUBTASK/doe_epymarl-main/results/gfootball/Time/training_ckpts
            
            # "results/models/{}".format(unique_token)
            os.makedirs(save_path, exist_ok=True)
            logger.console_logger.info("Saving models to {}".format(save_path))

            # learner should handle saving/loading -- delegate actor save/load to mac,
            # use appropriate filenames to do critics, optimizer states
            learner.save_models(save_path)   # 这是训练过程的中间ckpt

            if args.use_wandb and args.wandb_save_model:
                wandb_save_dir = os.path.join(
                    logger.wandb.dir, "models", args.unique_token, str(runner.t_env)
                )
                os.makedirs(wandb_save_dir, exist_ok=True)
                for f in os.listdir(save_path):
                    shutil.copyfile(
                        os.path.join(save_path, f), os.path.join(wandb_save_dir, f)
                    )

        episode += args.batch_size_run

        if (runner.t_env - last_log_T) >= args.log_interval:
            logger.log_stat("episode", episode, runner.t_env)
            logger.print_recent_stats()
            last_log_T = runner.t_env

    """ Save training final models """
    learner.save_models(layer_tmp_dir)   # 这是训练最终ckpt
    logger.console_logger.info(f"Save final model ckpt to {layer_tmp_dir}")
    
    """ Save buffers for DoE Classifier """
    if args.save_buffer:
        # 创建在训练结束时存储buffer用于DoE的路径
        # buffer_save_path = layer_tmp_dir + new exp name
        # buffer_save_path = os.path.join(dirname(dirname(abspath(__file__))), args.local_results_path, "buffers", args.env, args.time_stamp)
        # buffer_save_path = layer_tmp_dir
        # os.makedirs(layer_tmp_dir, exist_ok=True)

        """名字需要重新考虑 group id 方便后续 merge"""
        save_buffer_file_name = f'buffer_layer{args.layer_id}_decomposition{args.decomposition_id}_subtask{args.group_id}_iter{args.iter_id}_sample{args.sample_id}.pt'

        # 将本stage的buffer存储名字传入 doe cls cfg，用于下面的新的cls训练
        args.doe_classifier_cfg["save_buffer_file_name"] = save_buffer_file_name
        save_buffer_file_path = os.path.join(layer_tmp_dir, save_buffer_file_name)

        buffer_data = copy.deepcopy(buffer.data)
        # """默认的ia2c_ns中，obs_agent_id为False，creat task中手动设置为True"""
        # if args.obs_agent_id:
        #     buffer_data = process_buffer_for_doe(buffer_data, args.n_agents)

        th.save(buffer_data, save_buffer_file_path)
        logger.console_logger.info(f"Save buffer to {layer_tmp_dir} for DoE Classifier")
        # 目前在from config train中，写死的buffer名字为 load bufferpath+buffer.pt，需要改命名

    """ Train and Save DoE Classifier """
    # 直接用上面 buffer save path curr 的位置的 buffer_id.pt 来train
    # 这里要考虑一下merge以后的team doe cls，分配新的role_list
    if args.save_doe_cls:
        # 训练结束，创建属于本subtask的role_ids用于训练cls, group id 作为本组role_ids的label
        group_id = args.group_id
        curr_role_ids = {f"goal_{group_id}": list(range(args.n_agents)) }
        args.doe_classifier_cfg["role_ids"] = curr_role_ids

        
        # (0, ('goal_5', ['a' b c d e])) 对应 role list [0 0 0 0 0]
        # (0, ('defence', ['alice', 'bob'])) 和 (1, ('attack', ['carol', 'dave']))

        doe_classifier = doe_classifier_config_loader(
            n_agents=args.n_agents,
            cfg=args.doe_classifier_cfg,  # 本来是args.get("doe_classifier_cfg")，这里args是namespace形式
            buffer_path = layer_tmp_dir, # 使用当前保存的 buffer dir
            load_mode='train'
        )
        # from config设置了，如果有save cls，就会按照save name保存cls
        logger.console_logger.info(f"Save new doe cls to {layer_tmp_dir} for DoE Classifier")


    runner.close_env()
    logger.console_logger.info("Finished Training")


def args_sanity_check(config, _log):
    # set CUDA flags
    # config["use_cuda"] = True # Use cuda whenever possible!
    if config["use_cuda"] and not th.cuda.is_available():
        config["use_cuda"] = False
        _log.warning(
            "CUDA flag use_cuda was switched OFF automatically because no CUDA devices are available!"
        )

    if config["test_nepisode"] < config["batch_size_run"]:
        config["test_nepisode"] = config["batch_size_run"]
    else:
        config["test_nepisode"] = (
            config["test_nepisode"] // config["batch_size_run"]
        ) * config["batch_size_run"]

    return config

# 更新:由于runner的逻辑，buffer中不包含onehot，只在select action环节通过build input创建临时onehot，不需要剔除
# 剔除obs末尾的onehot编码，用于doe cls训练
def process_buffer_for_doe(buffer_data, total_agents=5):
    """处理buffer数据，去除onehot部分"""
    obs_data = buffer_data.transition_data["obs"]
    # 固定长度onehot
    fixed_len  = total_agents
    assert obs_data.shape[-1] > 115
    obs_without_onehot = obs_data[..., :-fixed_len]
    buffer_data.transition_data["obs"] = obs_without_onehot
    return buffer_data

