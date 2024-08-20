"""
   Trains a new DiT model.
   """
assert torch.cuda.is_available(), "Training currently requires at least one GPU."

# Setup accelerator:
accelerator = Accelerator()
device = accelerator.device

# Setup an experiment folder:
if accelerator.is_main_process:
    os.makedirs(args.results_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
    experiment_index = len(glob(f"{args.results_dir}/*"))
    model_string_name = args.model.replace("/", "-")  # e.g., DiT-XL/2 --> DiT-XL-2 (for naming folders)
    experiment_dir = f"{args.results_dir}/{experiment_index:03d}-{model_string_name}"  # Create an experiment folder
    checkpoint_dir = f"{experiment_dir}/checkpoints"  # Stores saved model checkpoints
    os.makedirs(checkpoint_dir, exist_ok=True)
    logger = create_logger(experiment_dir)
    logger.info(f"Experiment directory created at {experiment_dir}")

# Create model:
assert args.image_size % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
latent_size = args.image_size // 8
model = DiT_models[args.model](
    input_size=latent_size,
    num_classes=args.num_classes
)
# Note that parameter initialization is done within the DiT constructor
model = model.to(device)
ema = deepcopy(model).to(device)  # Create an EMA of the model for use after training
requires_grad(ema, False)
diffusion = create_diffusion(timestep_respacing="")  # default: 1000 steps, linear noise schedule
vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)
if accelerator.is_main_process:
    logger.info(f"DiT Parameters: {sum(p.numel() for p in model.parameters()):,}")

# Setup optimizer (we used default Adam betas=(0.9, 0.999) and a constant learning rate of 1e-4 in our paper):
opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0)

# Setup data:
features_dir = f"{args.feature_path}/imagenet256_features"
labels_dir = f"{args.feature_path}/imagenet256_labels"
dataset = CustomDataset(features_dir, labels_dir)
loader = DataLoader(
    dataset,
    batch_size=int(args.global_batch_size // accelerator.num_processes),
    shuffle=True,
    num_workers=args.num_workers,
    pin_memory=True,
    drop_last=True
)
if accelerator.is_main_process:
    logger.info(f"Dataset contains {len(dataset):,} images ({args.feature_path})")

# Prepare models for training:
update_ema(ema, model, decay=0)  # Ensure EMA is initialized with synced weights
model.train()  # important! This enables embedding dropout for classifier-free guidance
ema.eval()  # EMA model should always be in eval mode
model, opt, loader = accelerator.prepare(model, opt, loader)

# Variables for monitoring/logging purposes:
train_steps = 0
log_steps = 0
running_loss = 0
start_time = time()

if accelerator.is_main_process:
    logger.info(f"Training for {args.epochs} epochs...")
for epoch in range(args.epochs):
    if accelerator.is_main_process:
        logger.info(f"Beginning epoch {epoch}...")
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        x = x.squeeze(dim=1)
        y = y.squeeze(dim=1)
        t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=device)
        model_kwargs = dict(y=y)
        loss_dict = diffusion.training_losses(model, x, t, model_kwargs)
        loss = loss_dict["loss"].mean()
        opt.zero_grad()
        accelerator.backward(loss)
        opt.step()
        update_ema(ema, model)

        # Log loss values:
        running_loss += loss.item()
        log_steps += 1
        train_steps += 1
        if train_steps % args.log_every == 0:
            # Measure training speed:
            torch.cuda.synchronize()
            end_time = time()
            steps_per_sec = log_steps / (end_time - start_time)
            # Reduce loss history over all processes:
            avg_loss = torch.tensor(running_loss / log_steps, device=device)
            avg_loss = avg_loss.item() / accelerator.num_processes
            if accelerator.is_main_process:
                logger.info(
                    f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}")
            # Reset monitoring variables:
            running_loss = 0
            log_steps = 0
            start_time = time()

        # Save DiT checkpoint:
        if train_steps % args.ckpt_every == 0 and train_steps > 0:
            if accelerator.is_main_process:
                checkpoint = {
                    "model": model.module.state_dict(),
                    "ema": ema.state_dict(),
                    "opt": opt.state_dict(),
                    "args": args
                }
                checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                torch.save(checkpoint, checkpoint_path)
                logger.info(f"Saved checkpoint to {checkpoint_path}")

model.eval()  # important! This disables randomized embedding dropout
# do any sampling/FID calculation/etc. with ema (or model) in eval mode ...

if accelerator.is_main_process:
    logger.info("Done!")
