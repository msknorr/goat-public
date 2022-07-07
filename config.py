import torch

class Config:
    device = torch.device("cuda")
    resize = 512
    num_workers = 6

    # inference config
    conf_threshold = 0.9
    model_weights = "./weights/best-checkpoint-114epoch.bin"
    generate_plots = True

    # train config
    folder = '/weights'
    continue_path = None
    batch_size = 3
    n_epochs = 250
    lr = 0.0001
    accumulation_steps = 1  # gradient accumulation, 1 is off
    restrict_nr = 600  # if gpu OOM, reduce this to drop images with over N organoids

    verbose = True
    step_scheduler = False  # do scheduler.step after optimizer.step
    validation_scheduler = True  # do scheduler.step after validation stage loss

    SchedulerClass = torch.optim.lr_scheduler.ReduceLROnPlateau
    scheduler_params = dict(
        mode='min',
        factor=0.5,
        patience=5,
        verbose=False,
        threshold=0.00001,
        threshold_mode='abs',
        cooldown=0,
        min_lr=1e-5,
        eps=1e-08
    )
