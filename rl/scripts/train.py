if __name__ == '__main__':
    import os
    import argparse
    import time
    import datetime
    import torch_ac
    from torch.utils import tensorboard
    from torch.optim.lr_scheduler import StepLR

    import sys

    import utils
    from utils import device
    from model import ACModel
    from gym_minigrid.wrappers import currentAndGoalLocationObsWrapper


    # Parse arguments

    parser = argparse.ArgumentParser()

    ## General parameters
    parser.add_argument("--algo", required=True,
                        help="algorithm to use: a2c | ppo (REQUIRED)")
    parser.add_argument("--env", required=True,
                        help="name of the environment to train on (REQUIRED)")
    parser.add_argument("--model", default=None,
                        help="name of the model (default: {ENV}_{ALGO}_{TIME})")
    parser.add_argument("--model-type", default=None,
                        help="type of model to use (default from model.py)")
    parser.add_argument("--seed", type=int, default=1,
                        help="random seed (default: 1)")
    parser.add_argument("--log-interval", type=int, default=1,
                        help="number of updates between two logs (default: 1)")
    parser.add_argument("--save-interval", type=int, default=10,
                        help="number of updates between two saves (default: 10, 0 means no saving)")
    parser.add_argument("--procs", type=int, default=16,
                        help="number of processes (default: 16)")
    parser.add_argument("--frames", type=int, default=10**7,
                        help="number of frames of training (default: 1e7)")

    ## Parameters for main algorithm
    parser.add_argument("--epochs", type=int, default=4,
                        help="number of epochs for PPO (default: 4)")
    parser.add_argument("--batch-size", type=int, default=256,
                        help="batch size for PPO (default: 256)")
    parser.add_argument("--frames-per-proc", type=int, default=None,
                        help="number of frames per process before update (default: 5 for A2C and 128 for PPO)")
    parser.add_argument("--discount", type=float, default=0.99,
                        help="discount factor (default: 0.99)")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="learning rate (default: 0.001)")
    parser.add_argument("--gae-lambda", type=float, default=0.95,
                        help="lambda coefficient in GAE formula (default: 0.95, 1 means no gae)")
    parser.add_argument("--entropy-coef", type=float, default=0.01,
                        help="entropy term coefficient (default: 0.01)")
    parser.add_argument("--value-loss-coef", type=float, default=0.5,
                        help="value loss term coefficient (default: 0.5)")
    parser.add_argument("--max-grad-norm", type=float, default=0.5,
                        help="maximum norm of gradient (default: 0.5)")
    parser.add_argument("--optim-eps", type=float, default=1e-8,
                        help="Adam and RMSprop optimizer epsilon (default: 1e-8)")
    parser.add_argument("--optim-alpha", type=float, default=0.99,
                        help="RMSprop optimizer alpha (default: 0.99)")
    parser.add_argument("--clip-eps", type=float, default=0.2,
                        help="clipping epsilon for PPO (default: 0.2)")
    parser.add_argument("--recurrence", type=int, default=1,
                        help="number of time-steps gradient is backpropagated (default: 1). If > 1, a LSTM is added to the model to have memory.")
    parser.add_argument("--text", action="store_true", default=False,
                        help="add a GRU to the model to handle text input")
    parser.add_argument("--transition-as-input", action="store_true", default=False,
                        help="Give previous reward and previous action as input to the model. The environment must support reward and action as osbervation.")
    parser.add_argument("--linear-rnn", action="store_true", default=False,
                        help="Replace the rnn_memory with a linear version")
    parser.add_argument("--scheduler-patience", type=int, default=93,
                        help="Number of updates before the scheduler decreases the learning rate. lr=lr*0.99")
    parser.add_argument("--memory-weight-decay", type=float, default=0.0,
                        help="Add weight decay to rnn memory layers (default: 0.0)")
    parser.add_argument("--other-weight-decay", type=float, default=0.0,
                        help="Add weight decay to other layers,  e.g. cnn and policy network (default: 0.0)")
    args = parser.parse_args()

    args.rnn_mem = args.recurrence > 1

    # Set run dir
    date = datetime.datetime.now().strftime("%y-%m-%d-%H-%M-%S")
    default_model_name = f"{args.env}_{args.algo}_seed{args.seed}_{date}"

    model_name = args.model or default_model_name
    model_dir = utils.get_model_dir(model_name)

    # Load loggers and Tensorboard writer
    txt_logger = utils.get_txt_logger(model_dir)
    csv_file, csv_logger = utils.get_csv_logger(model_dir)
    tb_writer = tensorboard.SummaryWriter(model_dir)

    # Log command and all script arguments
    txt_logger.info("{}\n".format(" ".join(sys.argv)))
    txt_logger.info("{}\n".format(args))

    # Set seed for all randomness sources
    utils.seed(args.seed)

    # Set device
    txt_logger.info(f"Device: {device}\n")

    # Load environments
    envs = []
    for i in range(args.procs):
        if 'MemoryGame' in args.env:
            envs.append(currentAndGoalLocationObsWrapper(utils.make_env(args.env, args.seed + 10000 * i)))
        else:
            envs.append(utils.make_env(args.env, args.seed + 10000 * i))
    txt_logger.info("Environments loaded\n")

    # Load training status
    try:
        save_path = os.path.join(model_dir, "status")
        last_ckpt = 0
        while os.path.exists(save_path + str(last_ckpt) + '.pt'):
            last_ckpt += 1
        if last_ckpt != 0:
            last_ckpt -= 1
        status = utils.get_status(model_dir, last_ckpt)
        print(f'Loaded ckpt {last_ckpt}')
    except OSError:
        status = {"num_frames": 0, "update": 0}
    txt_logger.info("Training status loaded\n")

    # Load observations preprocessor
    obs_space, preprocess_obss = utils.get_obss_preprocessor(envs[0].observation_space)
    if "vocab" in status:
        preprocess_obss.vocab.load_vocab(status["vocab"])
    txt_logger.info("Observations preprocessor loaded")

    # Load model
    acmodel = ACModel(obs_space, envs[0].action_space, args.rnn_mem, args.text, args.transition_as_input, nb_tasks=envs[0].changing_reward, linear_rnn=args.linear_rnn)
    if "model_state" in status:
        acmodel.load_state_dict(status["model_state"])
    acmodel.to(device)
    txt_logger.info("Model loaded\n")
    txt_logger.info("{}\n".format(acmodel))

    # Load algo
    if args.algo == "a2c":
        algo = torch_ac.A2CAlgo(envs, acmodel, device, args.frames_per_proc, args.discount, args.lr, args.gae_lambda,
                                args.entropy_coef, args.value_loss_coef, args.max_grad_norm, args.recurrence,
                                args.optim_alpha, args.optim_eps, preprocess_obss)
    elif args.algo == "ppo":
        algo = torch_ac.PPOAlgo(envs, acmodel, device, args.frames_per_proc, args.discount, args.lr, args.gae_lambda,
                                args.entropy_coef, args.value_loss_coef, args.max_grad_norm, args.recurrence,
                                args.optim_eps, args.clip_eps, args.epochs, args.batch_size, preprocess_obss,
                                memory_weight_decay=args.memory_weight_decay, other_weight_decay=args.other_weight_decay)
    else:
        raise ValueError("Incorrect algorithm name: {}".format(args.algo))

    scheduler = StepLR(algo.optimizer, step_size=args.scheduler_patience, gamma=0.99) # every 93 updates: lr=lr*0.99


    if "optimizer_state" in status:
        algo.optimizer.load_state_dict(status["optimizer_state"])
    txt_logger.info("Optimizer loaded\n")

    # Train model
    num_frames = status["num_frames"]
    update = status["update"]
    optimal_choice = []
    incomplete_trials = []
    avg_nb_trial_per_episode = []
    start_time = time.time()

    # Save untrained checkpoint
    if status["num_frames"] == 0:
        status = {"num_frames": num_frames, "update": update,
                  "model_state": acmodel.state_dict(), "optimizer_state": algo.optimizer.state_dict()}
        if hasattr(preprocess_obss, "vocab"):
            status["vocab"] = preprocess_obss.vocab.vocab
        utils.save_status(status, model_dir)

    while num_frames < args.frames:
        # Update model parameters
        update_start_time = time.time()
        exps, logs1 = algo.collect_experiences()
        logs2 = algo.update_parameters(exps)
        logs = {**logs1, **logs2}
        update_end_time = time.time()

        scheduler.step()

        num_frames += logs["num_frames"]
        update += 1
        optimal_choice.extend(logs["optimal_choice"])
        incomplete_trials.extend(logs['incomplete_trials'])
        avg_nb_trial_per_episode.extend(logs['average_nb_trial_per_episode'])

        # Print logs

        if update % args.log_interval == 0:
            fps = logs["num_frames"]/(update_end_time - update_start_time)
            if len(optimal_choice) > 0:
                optimal_percentage = (sum(optimal_choice) / len(optimal_choice))*100
            else:
                optimal_percentage = 0
            if len(incomplete_trials) > 0:
                incomplete_trials_percentage = (sum(incomplete_trials) / len(incomplete_trials))*100
            else:
                incomplete_trials_percentage = 0
            duration = int(time.time() - start_time)
            return_per_episode = utils.synthesize(logs["return_per_episode"])
            rreturn_per_episode = utils.synthesize(logs["reshaped_return_per_episode"])
            num_frames_per_episode = utils.synthesize(logs["num_frames_per_episode"])

            header = ["update", "frames", "FPS", "duration", "perc_optimal_choice", "num_choice", "perc_incomplete_trials", "num_trials", 'avg_num_trial_per_episode']
            data = [update, num_frames, fps, duration, optimal_percentage, len(optimal_choice), incomplete_trials_percentage, len(incomplete_trials), sum(avg_nb_trial_per_episode)/len(avg_nb_trial_per_episode)]
            header += ["rreturn_" + key for key in rreturn_per_episode.keys()]
            data += rreturn_per_episode.values()
            header += ["num_frames_" + key for key in num_frames_per_episode.keys()]
            data += num_frames_per_episode.values()
            header += ["entropy", "value", "policy_loss", "value_loss", "grad_norm"]
            data += [logs["entropy"], logs["value"], logs["policy_loss"], logs["value_loss"], logs["grad_norm"]]
            header += ['learning_rate']
            data += scheduler.get_last_lr()

            txt_logger.info(
                "U {} | F {:06} | FPS {:04.0f} | D {} | O: {:.2f}% of {} | IT: {:.2f}% of {} | TpE {:.0f} | rR:μσmM {:.2f} {:.2f} {:.2f} {:.2f} | F:μσmM {:.1f} {:.1f} {} {} | H {:.3f} | V {:.3f} | pL {:.3f} | vL {:.3f} | ∇ {:.3f} | lr {}"
                .format(*data))

            header += ["return_" + key for key in return_per_episode.keys()]
            data += return_per_episode.values()

            if status["num_frames"] == 0:
                csv_logger.writerow(header)
            csv_logger.writerow(data)
            csv_file.flush()

            for field, value in zip(header, data):
                tb_writer.add_scalar(field, value, num_frames)

            optimal_choice = []
            incomplete_trials = []
            avg_nb_trial_per_episode = []

        # Save status

        if args.save_interval > 0 and update % args.save_interval == 0:
            status = {"num_frames": num_frames, "update": update,
                      "model_state": acmodel.state_dict(), "optimizer_state": algo.optimizer.state_dict()}
            if hasattr(preprocess_obss, "vocab"):
                status["vocab"] = preprocess_obss.vocab.vocab
            utils.save_status(status, model_dir)
            txt_logger.info("Status saved")
