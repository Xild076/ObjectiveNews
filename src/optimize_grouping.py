import optuna
from grouping import cluster_texts, load_samples, reward_function
import warnings

def objective(trial):
    weights = trial.suggest_float("weights", 0.0, 1.0)
    context = trial.suggest_categorical("context", [True, False])
    context_len = trial.suggest_int("context_len", 1, 20)
    preprocess = trial.suggest_categorical("preprocess", [True, False])
    norm = trial.suggest_categorical("norm", ['l1', 'l2', 'max', 'none'])
    n_neighbors = trial.suggest_int("n_neighbors", 5, 50)
    n_components = trial.suggest_int("n_components", 2, 10)
    umap_metric = trial.suggest_categorical("umap_metric", ['euclidean', 'manhattan', 'cosine', 'correlation', 'chebyshev'])
    cluster_metric = trial.suggest_categorical("cluster_metric", ['euclidean', 'manhattan', 'cosine'])
    algorithm = trial.suggest_categorical("algorithm", ['best', 'generic', 'prims_kdtree', 'boruvka_kdtree'])
    cluster_selection_method = trial.suggest_categorical("cluster_selection_method", ['eom', 'leaf'])

    try:
        samples = load_samples()

        total_reward = 0
        for sample in samples:
            sentences = sample['sentences']
            clusters_true = sample['clusters']
            
            filtered_data = [(s, c) for s, c in zip(sentences, clusters_true) if s.strip()]
            if not filtered_data:
                continue
                
            sentences, clusters_true = zip(*filtered_data)
            sentences = list(sentences)
            clusters_true = list(clusters_true)
            
            if len(sentences) == 0:
                continue
                
            clusters_pred = cluster_texts(
                sentences,
                weights=weights,
                context=context,
                context_len=context_len,
                preprocess=preprocess,
                attention=False,
                norm=norm,
                n_neighbors=n_neighbors,
                n_components=n_components,
                umap_metric=umap_metric,
                cluster_metric=cluster_metric,
                algorithm=algorithm,
                cluster_selection_method=cluster_selection_method
            )
            
            if not clusters_pred or len(clusters_pred) != len(clusters_true):
                continue
                
            reward = reward_function(clusters_pred, clusters_true)
            total_reward += reward

        return total_reward / len(samples) if samples else 0
    except Exception as e:
        return 0

study = optuna.create_study(direction="maximize")
try:
    study.optimize(objective, n_trials=1000)
except KeyboardInterrupt:
    print("Optimization interrupted by user")
except Exception as e:
    print(f"Optimization failed: {e}")

if study.best_trial is not None:
    best_trial = study.best_trial
    print("Best trial:")
    print(f"  Value: {best_trial.value}")
    print("  Params:")
    for key, value in best_trial.params.items():
        print(f"    {key}: {value}")
else:
    print("No successful trials found")


# {'weights': 0.33406325713438734, 'context': True, 'context_len': 2, 'preprocess': True, 'norm': 'l2', 'n_neighbors': 6, 'n_components': 4, 'umap_metric': 'manhattan', 'cluster_metric': 'euclidean', 'algorithm': 'boruvka_kdtree', 'cluster_selection_method': 'leaf'}. Best is trial 578 with value: 0.5986058381220961.