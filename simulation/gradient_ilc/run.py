import init
import time


def main(args):
    import torch

    torch.manual_seed(args.seed_torch)
    rng = torch.Generator().manual_seed(args.seed_torch)

    rhm_data = init.init_dataset(args)
    train_dataloader, test_dataloader, val_dataloader, train_eval_dataloader = (
        init.init_data_loaders(args, rhm_data)
    )
    print(
        "Initialized datasets: train size %d, test size %d, val size %d"
        % (len(rhm_data["train"]), len(rhm_data["test"]), len(rhm_data["val"])),
        flush=True,
    )

    import model

    resumption_state = (
        torch.load(args.resume_checkpoint_path, weights_only=True)
        if args.resume
        else None
    )
    student = model.init_model(args, resumption_state=resumption_state)
    teacher = model.init_teacher(student, args, resumption_state=resumption_state)
    print(
        "Initialized models with device:",
        args.device,
        " student device:",
        next(student.parameters()).device,
        " teacher device:",
        next(teacher.parameters()).device if teacher is not None else None,
    )

    import train

    optimizer = train.init_optimizer(args, student, resumption_state=resumption_state)

    import observables

    observer_state = (
        torch.load(args.resume_observables_path, weights_only=False)
        if args.resume
        else None
    )
    observer = observables.init_observer(
        len(train_dataloader),
        train_eval_dataloader,
        test_dataloader,
        val_dataloader,
        args,
        observer_state=observer_state,
        rhm_data=rhm_data,
    )
    print("Initialized observer.", flush=True)

    starting_step = 0
    if args.resume:
        tms = resumption_state["train_model_state"]
        starting_epoch = tms["epoch"]
        starting_step = tms["step"]
        observer.clear_samples_after_step(starting_step)
        last_epoch = starting_epoch
    else:
        epoch, step = 0, 0
        observer.observe(epoch, step, student, teacher)
        last_epoch = 0

    training_generator = train.train_model(
        student,
        optimizer,
        train_dataloader,
        args.Nepoch,
        teacher=teacher,
        ema_alpha=args.ema_alpha,
        starting_step=starting_step,
        use_autojac=args.use_autojac,
        rhm_data=rhm_data,
        enable_probes=args.enable_probes,
    )
    training_started = time.monotonic()
    for train_model_state in training_generator:
        epoch, step = train_model_state["epoch"], train_model_state["step"]
        if (
            epoch in observer.sample_epochs and last_epoch != epoch
        ) or step in observer.sample_steps:
            observer.observe(epoch, step, student, teacher)
            observer.save()
        if step in observer.checkpoint_steps:
            observer.checkpoint(student, teacher, optimizer, train_model_state)
        last_epoch = epoch
        if args.max_training_seconds > 0 and time.monotonic() - training_started >= args.max_training_seconds:
            print(f"Graceful training cutoff at step {step}", flush=True)
            break

    if step not in observer.sample_steps:
        observer.observe(epoch, step, student, teacher)
        observer.save()
    if step not in observer.checkpoint_steps:
        observer.checkpoint(student, teacher, optimizer, train_model_state)
    print("Finished run on step/epoch:", step, "/", epoch, flush=True)
    return observer, student, teacher


if __name__ == "__main__":
    parser = init.init_parser()
    args = parser.parse_args()
    main(args)
