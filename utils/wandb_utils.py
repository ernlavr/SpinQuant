import wandb

def start_sweep(sweep_config, main_func):
    sweep_id = wandb.sweep(sweep_config, project='spinquant-noise')
    def sweep_main():
        with wandb.init(settings=wandb.Settings(console='wrap')) as run:
            config = wandb.config
            # Here you can access config parameters and use them in your training
            print(f"Running training with config: {config}")
            main_func(config)
    wandb.agent(sweep_id, function=sweep_main)
    
def start_run(run_name, config, main_func):
    with wandb.init(project='spinquant-noise', 
                    name=run_name, 
                    config=config, 
                    settings=wandb.Settings(console='wrap')) as run:
        main_func()
        
    