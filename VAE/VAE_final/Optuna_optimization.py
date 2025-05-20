import optuna
from Model_architecture import VAE_Seed()
from Train import VAE_train()
import numpy as np
from Prognostic_criteria import fitness()


def optimize_hyperparameters_optuna(
    num_features, target_rows, vae_train_data, vae_val_data, vae_test_data,
    n_trials=40,
    direction='minimize' , 
):
    """
    Optimize VAE hyperparameters using Optuna's TPE sampler and pruning.
    Returns best_params dict and best_value.
    """
    def objective(trial):
        # Suggest hyperparameters
        hidden_1     = trial.suggest_int('hidden_1',     40, 120)
        learning_rate= trial.suggest_loguniform('learning_rate', 1e-3, 1e-2)
        epochs       = trial.suggest_int('epochs',       500, 1000)
        reloss_coeff = trial.suggest_uniform('reloss_coeff', 0.05, 0.6)
        klloss_coeff = trial.suggest_uniform('klloss_coeff', 1.4, 1.8)
        moloss_coeff = trial.suggest_uniform('moloss_coeff', 2.6, 4.0)
        
        #these are added hyperparameters, these can be removed if necessary
        hidden_2 = trial.suggest_int('hidden_2',8,32)
        batch_size = trial.suggest_int('batch_size',100,1000)

        # Train VAE with these params
        hi_train, hi_test, hi_val, vae, epoch_losses, losses = VAE_train(target_rows,
            vae_train_data, vae_val_data, vae_test_data,
            hidden_1, batch_size, learning_rate, epochs,
            reloss_coeff, klloss_coeff, moloss_coeff,
            num_features, hidden_2=hidden_2,
            )

        # Compute fitness error on stacked health indicators
        hi_all = np.vstack((hi_train, hi_test, hi_val))
        _, _, _, _, error = fitness(hi_all)
        trial.report(error, step=0)
        return error

    # Create study with TPE sampler and median pruner
    sampler = optuna.samplers.TPESampler(seed=VAE_Seed.vae_seed)
    pruner  = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1)
    study   = optuna.create_study(direction=direction, sampler=sampler, pruner=pruner)
    study.optimize(objective, n_trials=n_trials, timeout=None)

    print("Best trial:", study.best_trial.params)
    return study.best_trial.params, study.best_value