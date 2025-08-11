import optuna
from tqdm import tqdm
import os
import sys
try:
    from .grouping import cluster_sentences, load_samples, reward_function, SelfAttentionModel, cluster_texts
    from ..utility import load_sent_transformer, DLog, SentenceHolder
except Exception:
    try:
        from src.grouping.grouping import cluster_sentences, load_samples, reward_function, SelfAttentionModel, cluster_texts
        from src.utility import load_sent_transformer, DLog, SentenceHolder
    except Exception:
        _CUR = os.path.dirname(__file__)
        _PARENT = os.path.abspath(os.path.join(_CUR, '..'))
        _ROOT = os.path.abspath(os.path.join(_CUR, '..', '..'))
        for _p in (_PARENT, _ROOT):
            if _p not in sys.path:
                sys.path.insert(0, _p)
        from grouping.grouping import cluster_sentences, load_samples, reward_function, SelfAttentionModel, cluster_texts
        from utility import load_sent_transformer, DLog, SentenceHolder
import csv
import torch
import torch.nn as nn
import random
import numpy as np
import torch.optim as optim
from torch.distributions import Normal
import copy

logger = DLog("OPTIMIZE_GROUPING")

class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin
    
    def forward(self, anchor, positive, negative):
        pos_dist = torch.norm(anchor - positive, dim=1)
        neg_dist = torch.norm(anchor - negative, dim=1)
        loss = torch.clamp(pos_dist - neg_dist + self.margin, min=0)
        return loss.mean()

class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim=256):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
    def forward(self, x):
        return torch.sigmoid(self.network(x)) * 1.2 + 0.4

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin
    
    def forward(self, embeddings, labels):
        loss = 0
        count = 0
        
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                distance = torch.norm(embeddings[i] - embeddings[j])
                
                if labels[i] == labels[j]:
                    loss += distance ** 2
                else:
                    loss += torch.clamp(self.margin - distance, min=0) ** 2
                count += 1
        
        return loss / count if count > 0 else loss

class DifferentiableClusteringLoss(nn.Module):
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, embeddings, target_clusters):
        device = embeddings.device
        similarities = torch.mm(embeddings, embeddings.t())
        similarities = similarities / self.temperature
        
        target_clusters = torch.tensor(target_clusters, dtype=torch.long, device=device)
        target_sim = (target_clusters.unsqueeze(0) == target_clusters.unsqueeze(1)).float()
        
        prob_matrix = torch.softmax(similarities, dim=1)
        loss = nn.BCELoss()(prob_matrix, target_sim)
        
        return loss

def objective(trial):
    logger.info("Loading objective...")
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
            sentences, clusters_true = sample['sentences'], sample['clusters']
            filtered_data = [(s, c) for s, c in zip(sentences, clusters_true) if s.strip()]
            if not filtered_data: continue
            sentences, clusters_true = zip(*filtered_data)
            if not sentences: continue
            clusters_pred = cluster_sentences(list(sentences), 
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
                                          cluster_selection_method=cluster_selection_method)
            if not clusters_pred or len(clusters_pred) != len(clusters_true): continue
            total_reward += reward_function(clusters_pred, list(clusters_true))
        return total_reward / len(samples) if samples else 0
    except Exception as e:
        logger.warning(f"Error in objective: {str(e)[:50]}")
        return 0

def log_best_trial_to_csv(study, trial):
    logger.info("Saving best trial to csv...")
    if trial.state != optuna.trial.TrialState.COMPLETE or study.best_trial.number != trial.number:
        return
    filepath = "optuna_best_trials_log.csv"
    fieldnames = list(trial.params.keys()) + ["value"]
    file_exists = os.path.isfile(filepath)
    with open(filepath, mode="a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow({**trial.params, "value": trial.value})

def optuna_optimization():
    logger.info("Optuna optimizing...")
    study = optuna.create_study(direction="maximize")
    try:
        study.optimize(objective, n_trials=20000, callbacks=[log_best_trial_to_csv])
    except KeyboardInterrupt:
        print("Optimization interrupted by user.")
    if study.best_trial:
        print("Best trial:", study.best_trial.params, "Value:", study.best_trial.value)

def compute_clustering_reward_loss(embeddings, clusters_true, device):
    logger.info("Computing clustering reward...")
    cluster_map = {}
    for i, cluster_id in enumerate(clusters_true):
        if cluster_id not in cluster_map:
            cluster_map[cluster_id] = []
        cluster_map[cluster_id].append(i)
    
    if len(cluster_map) < 2:
        return torch.tensor(0.0, device=device)
    
    intra_distances = []
    centroids = []
    
    for indices in cluster_map.values():
        if len(indices) < 2:
            continue
        cluster_embeddings = embeddings[indices]
        centroid = cluster_embeddings.mean(dim=0)
        centroids.append(centroid)
        distances = torch.norm(cluster_embeddings - centroid, dim=1)
        intra_distances.extend(distances)
    
    if len(centroids) < 2 or len(intra_distances) == 0:
        return torch.tensor(0.0, device=device)
    
    inter_distances = []
    for i in range(len(centroids)):
        for j in range(i + 1, len(centroids)):
            distance = torch.norm(centroids[i] - centroids[j])
            inter_distances.append(distance)
    
    if len(inter_distances) == 0:
        return torch.tensor(0.0, device=device)
    
    avg_intra = torch.stack(intra_distances).mean()
    avg_inter = torch.stack(inter_distances).mean()
    
    silhouette_score = (avg_inter - avg_intra) / (torch.max(avg_inter, avg_intra) + 1e-8)
    embedding_penalty = 0.01 * torch.norm(embeddings).mean()
    
    return -silhouette_score + embedding_penalty

def create_triplets(sample, num_triplets=80):
    sentences, clusters_true = sample['sentences'], sample['clusters']
    cluster_map = {}
    for i, cluster_id in enumerate(clusters_true):
        if cluster_id not in cluster_map:
            cluster_map[cluster_id] = []
        cluster_map[cluster_id].append(i)
    
    valid_clusters = [cid for cid, indices in cluster_map.items() if len(indices) >= 2]
    if len(valid_clusters) < 2:
        return []
    
    triplets = []
    for _ in range(num_triplets):
        anchor_cluster_id = random.choice(valid_clusters)
        anchor_idx, positive_idx = random.sample(cluster_map[anchor_cluster_id], 2)
        
        negative_cluster_id = random.choice([cid for cid in valid_clusters if cid != anchor_cluster_id])
        negative_idx = random.choice(cluster_map[negative_cluster_id])
        
        triplets.append((anchor_idx, positive_idx, negative_idx))
    
    return triplets

def cluster_from_embeddings(embeddings):
    if len(embeddings) <= 1:
        return [0] * len(embeddings)
    
    from sklearn.preprocessing import normalize
    import umap
    import hdbscan
    
    embeddings = normalize(embeddings, norm='l2')
    
    n_neighbors = min(10, len(embeddings) - 1)
    reducer = umap.UMAP(n_neighbors=n_neighbors, n_components=2, metric='cosine')
    reduced_embeddings = reducer.fit_transform(embeddings)
    
    min_cluster_size = max(2, len(embeddings) // 10)
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, metric='euclidean')
    return clusterer.fit_predict(reduced_embeddings).tolist()

def evolutionary_strategy_training(model_refiner, data, population_size=100, generations=200, mutation_std=0.06):
    device = next(model_refiner.parameters()).device
    
    def evaluate_individual(individual, model_refiner, data_subset):
        original_state = {name: param.clone() for name, param in model_refiner.named_parameters()}
        try:
            idx = 0
            for name, param in model_refiner.named_parameters():
                param_size = param.numel()
                param.data.copy_(torch.tensor(
                    individual[idx:idx + param_size].reshape(param.shape),
                    dtype=param.dtype, device=param.device
                ))
                idx += param_size
            
            total_reward = 0
            valid_count = 0
            
            with torch.no_grad():
                for sample in data_subset:
                    if len(sample['sentences']) < 3 or len(set(sample['clusters'])) < 2:
                        continue
                    try:
                        pred_clusters = cluster_sentences(sample['sentences'], att_model=model_refiner, attention=True)
                        if len(pred_clusters) == len(sample['clusters']):
                            total_reward += reward_function(pred_clusters, sample['clusters'])
                            valid_count += 1
                    except:
                        continue
            
            score = total_reward / max(valid_count, 1)
        except:
            score = -1.0
        finally:
            for name, param in model_refiner.named_parameters():
                param.data.copy_(original_state[name])
        
        return score
    
    def mutate(individual, rate):
        noise = np.random.normal(0, rate, len(individual))
        return individual + noise
    
    def exploration_mutate(individual, rate, exploration_factor=5.0):
        base_noise = np.random.normal(0, rate, len(individual))
        exploration_noise = np.random.normal(0, rate * exploration_factor, len(individual))
        exploration_mask = np.random.random(len(individual)) < 0.2
        noise = np.where(exploration_mask, exploration_noise, base_noise)
        return individual + noise
    
    model_params = np.concatenate([p.flatten().detach().cpu().numpy() for p in model_refiner.parameters()])
    model_params = model_params.astype(np.float32)
    
    data_subset = data[:min(25, len(data))]
    elite_count = max(15, population_size // 5)
    random_count = max(10, population_size // 10)
    
    population = []
    for i in range(population_size):
        if i < 8:
            noise_scale = 0.005
        else:
            noise_scale = mutation_std * np.random.exponential(0.6)
        individual = model_params + np.random.normal(0, noise_scale, len(model_params))
        population.append(individual.astype(np.float32))
    
    best_individual = model_params.copy()
    best_fitness = -float('inf')
    stagnation = 0
    adaptive_mutation = mutation_std
    elite_archive = []
    restart_count = 0
    
    for generation in range(generations):
        fitness_scores = []
        for individual in population:
            score = evaluate_individual(individual, model_refiner, data_subset)
            fitness_scores.append(score)
        
        fitness_array = np.array(fitness_scores)
        if np.all(fitness_array == fitness_array[0]):
            fitness_array += np.random.normal(0, 1e-6, len(fitness_array))
        
        sorted_indices = np.argsort(fitness_array)[::-1]
        current_best = fitness_array[sorted_indices[0]]
        
        if current_best > best_fitness:
            best_fitness = current_best
            best_individual = population[sorted_indices[0]].copy()
            stagnation = 0
            adaptive_mutation = min(mutation_std * 1.3, 0.2)
            
            if len(elite_archive) < 10:
                elite_archive.append(best_individual.copy())
            else:
                elite_archive[restart_count % 10] = best_individual.copy()
        else:
            stagnation += 1
            adaptive_mutation *= 0.96
        
        if stagnation > 10:
            adaptive_mutation *= 2.0
        
        if stagnation > 18:
            restart_count += 1
            print(f"Stagnation restart #{restart_count}, keeping top elites")
            
            new_population = []
            
            if len(elite_archive) >= 5:
                top_elites = elite_archive[:min(8, len(elite_archive))]
                new_population.extend([elite.copy() for elite in top_elites])
            else:
                top_performers = [population[i].copy() for i in sorted_indices[:8]]
                new_population.extend(top_performers)
            
            exploration_individuals = population_size - len(new_population)
            for _ in range(exploration_individuals):
                if len(new_population) > 0 and np.random.random() < 0.6:
                    base_individual = new_population[np.random.randint(len(new_population))]
                    new_individual = exploration_mutate(base_individual, adaptive_mutation * 3.0, exploration_factor=6.0)
                else:
                    new_individual = model_params + np.random.normal(0, adaptive_mutation * 5, len(model_params))
                    
                new_population.append(new_individual)
            
            population = new_population
            stagnation = 0
            adaptive_mutation = mutation_std * 2.5
            continue
        
        new_population = []
        elite_indices = sorted_indices[:elite_count]
        new_population.extend([population[i].copy() for i in elite_indices[:8]])
        
        for _ in range(population_size - len(new_population) - random_count):
            if np.random.random() < 0.8:
                parent1_idx = np.random.choice(elite_indices)
                parent2_idx = np.random.choice(elite_indices)
                alpha = np.random.beta(1.2, 1.2)
                child = alpha * population[parent1_idx] + (1 - alpha) * population[parent2_idx]
            else:
                parent_idx = np.random.choice(elite_indices)
                child = population[parent_idx].copy()
            
            if stagnation > 6:
                child = exploration_mutate(child, adaptive_mutation, exploration_factor=3.0)
            else:
                child = mutate(child, adaptive_mutation)
            new_population.append(child)
        
        for _ in range(random_count):
            if np.random.random() < 0.2:
                immigrant = model_params + np.random.normal(0, adaptive_mutation * 3.0, len(model_params))
            else:
                base_idx = np.random.choice(elite_indices)
                if stagnation > 6:
                    immigrant = exploration_mutate(population[base_idx], adaptive_mutation * 2.5, exploration_factor=4.0)
                else:
                    immigrant = mutate(population[base_idx], adaptive_mutation * 2.0)
            new_population.append(immigrant)
        
        population = new_population
        
        if generation % 1 == 0:
            trend = "↑" if stagnation == 0 else "↓"
            print(f"Gen {generation:3d}: Best: {best_fitness:.4f} {trend} | Mut: {adaptive_mutation:.3f} | Stag: {stagnation}")
        
        if stagnation > 30:
            break
    
    idx = 0
    for name, param in model_refiner.named_parameters():
        param_size = param.numel()
        param.data.copy_(torch.tensor(
            best_individual[idx:idx + param_size].reshape(param.shape),
            dtype=param.dtype, device=param.device
        ))
        idx += param_size
    
    return model_refiner

def train_with_policy_gradient(model_refiner, data, epochs=100):
    logger.info("Training with policy gradient...")
    model = load_sent_transformer()
    policy_net = PolicyNetwork(384).to(next(model_refiner.parameters()).device)
    optimizer = optim.AdamW(list(model_refiner.parameters()) + list(policy_net.parameters()), lr=0.001, weight_decay=1e-4)
    device = next(model_refiner.parameters()).device
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    best_reward = -float('inf')
    patience = 50
    no_improve = 0
    baseline = 0.0
    baseline_momentum = 0.95
    value_function = nn.Sequential(
        nn.Linear(384, 128),
        nn.ReLU(),
        nn.Linear(128, 1)
    ).to(device)
    value_optimizer = optim.AdamW(value_function.parameters(), lr=0.002)
    
    for epoch in range(epochs):
        total_reward = 0
        total_loss = 0
        batch_count = 0
        episode_rewards = []
        episode_values = []
        episode_log_probs = []
        
        random.shuffle(data)
        
        for sample in data[:min(len(data), 40)]:
            if len(sample['sentences']) < 3 or len(set(sample['clusters'])) < 2:
                continue
                
            sentences = sample['sentences']
            true_clusters = sample['clusters']
            
            embeddings = model.encode(sentences, show_progress_bar=False)
            embedding_tensor = torch.tensor(embeddings, dtype=torch.float32).to(device)
            context_embedding = embedding_tensor.mean(dim=0)
            
            with torch.no_grad():
                base_embeddings = model_refiner(embedding_tensor)
                base_clusters = cluster_from_embeddings(base_embeddings.cpu().numpy())
                if len(base_clusters) == len(true_clusters):
                    base_reward = reward_function(base_clusters, true_clusters)
                else:
                    base_reward = 0.0
            
            action_prob = policy_net(context_embedding)
            action_std = 0.1 + 0.05 * torch.sigmoid(action_prob)
            action_dist = Normal(action_prob, action_std)
            sampled_action = action_dist.sample()
            log_prob = action_dist.log_prob(sampled_action)
            
            scaling_factor = torch.clamp(sampled_action, 0.3, 1.8)
            refined_embeddings = model_refiner(embedding_tensor)
            scaled_embeddings = refined_embeddings * scaling_factor
            refined_embeddings_np = scaled_embeddings.detach().cpu().numpy()
            
            pred_clusters = cluster_from_embeddings(refined_embeddings_np)
            if len(pred_clusters) == len(true_clusters):
                reward = reward_function(pred_clusters, true_clusters)
                
                shaped_reward = reward - base_reward
                shaped_reward += 0.1 * (1.0 - abs(scaling_factor.item() - 1.0))
                
                state_value = value_function(context_embedding).squeeze()
                
                episode_rewards.append(shaped_reward)
                episode_values.append(state_value)
                episode_log_probs.append(log_prob)
                
                total_reward += reward
                batch_count += 1
        
        if len(episode_rewards) > 0:
            rewards_tensor = torch.tensor(episode_rewards, device=device)
            values_tensor = torch.stack(episode_values)
            
            baseline = baseline_momentum * baseline + (1 - baseline_momentum) * rewards_tensor.mean().item()
            
            advantages = rewards_tensor - values_tensor.detach()
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
            policy_loss = 0
            value_loss = 0
            entropy_loss = 0
            
            for i, (log_prob, advantage, reward, value) in enumerate(zip(episode_log_probs, advantages, rewards_tensor, values_tensor)):
                policy_loss -= log_prob * advantage
                value_loss += (reward - value) ** 2
            
            policy_loss = policy_loss / len(episode_rewards)
            value_loss = value_loss / len(episode_rewards)
            
            for i in range(min(len(episode_rewards), len(data))):
                sample_idx = i % len(data)
                sample_embeddings = model.encode(data[sample_idx]['sentences'], show_progress_bar=False)
                sample_context = torch.tensor(sample_embeddings, dtype=torch.float32).to(device).mean(dim=0)
                action_prob = policy_net(sample_context)
                action_std = 0.1 + 0.05 * torch.sigmoid(action_prob)
                entropy_loss -= torch.log(action_std + 1e-8)
            
            entropy_loss = entropy_loss / min(len(episode_rewards), len(data))
            total_loss_item = policy_loss + 0.5 * value_loss - 0.01 * entropy_loss
            
            optimizer.zero_grad()
            value_optimizer.zero_grad()
            total_loss_item.backward()
            
            torch.nn.utils.clip_grad_norm_(list(model_refiner.parameters()) + list(policy_net.parameters()), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(value_function.parameters(), max_norm=1.0)
            
            optimizer.step()
            value_optimizer.step()
            
            total_loss += total_loss_item.item()
        
        scheduler.step()
        
        if batch_count > 0:
            avg_reward = total_reward / batch_count
            avg_loss = total_loss
            
            if avg_reward > best_reward:
                best_reward = avg_reward
                no_improve = 0
            else:
                no_improve += 1
            
            if epoch % 2 == 0:
                print(f"Epoch {epoch}: Reward: {avg_reward:.4f}, Loss: {avg_loss:.4f}, Baseline: {baseline:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}")
            
            if no_improve >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

def train_with_differentiable_loss(model_refiner, data, val_data, epochs=100, best_params=None):
    logger.info("Testing with diff loss...")
    model = load_sent_transformer()
    criterion = DifferentiableClusteringLoss(temperature=0.07)
    optimizer = torch.optim.AdamW(model_refiner.parameters(), lr=0.003, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.005, epochs=epochs, steps_per_epoch=1)
    device = next(model_refiner.parameters()).device
    
    if best_params is None:
        best_params = {
            "weights": 0.4088, "context": True, "context_len": 2, "preprocess": False, 
            "norm": 'max', "n_neighbors": 6, "n_components": 4, "umap_metric": 'correlation', 
            "cluster_metric": 'euclidean', "algorithm": 'generic', "cluster_selection_method": 'leaf'
        }
    
    best_reward = -float('inf')
    patience = 20
    no_improve = 0
    
    for epoch in range(epochs):
        model_refiner.train()
        total_loss = 0
        batch_count = 0
        
        random.shuffle(data)
        
        for sample in data[:min(len(data), 50)]:
            if len(sample['sentences']) < 3 or len(set(sample['clusters'])) < 2:
                continue
                
            sentences = sample['sentences']
            true_clusters = sample['clusters']
            
            embeddings = model.encode(sentences, show_progress_bar=False)
            embedding_tensor = torch.tensor(embeddings, dtype=torch.float32).to(device)
            
            refined_embeddings = model_refiner(embedding_tensor)
            
            loss = criterion(refined_embeddings, true_clusters)
            regularization = 0.001 * torch.norm(refined_embeddings, p=2)
            total_loss_item = loss + regularization
            
            optimizer.zero_grad()
            total_loss_item.backward()
            torch.nn.utils.clip_grad_norm_(model_refiner.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += total_loss_item.item()
            batch_count += 1
        
        scheduler.step()
        
        if batch_count > 0:
            avg_reward = evaluate_model_on_validation(model_refiner, val_data, best_params, attention=True)
            avg_loss = total_loss / batch_count
            
            if epoch % 5 == 0:
                print(f"Epoch {epoch}: Loss: {avg_loss:.4f}, Reward: {avg_reward:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}")
            
            if avg_reward > best_reward:
                best_reward = avg_reward
                no_improve = 0
                
                checkpoint = {
                    'model_state_dict': model_refiner.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'epoch': epoch + 1,
                    'best_score': best_reward,
                    'best_params': best_params,
                    'training_method': 'differentiable'
                }
                save_best_model(checkpoint)
                if epoch % 5 == 0:
                    print(f"  New best model saved! Score: {best_reward:.4f}")
            else:
                no_improve += 1
                if no_improve >= patience:
                    print(f"Early stopping at epoch {epoch}")
                    break
    
    return best_reward

def train_contrastive(model_refiner, data, val_data, epochs=100, best_params=None):
    logger.info("Training with contrastive...")
    model = load_sent_transformer()
    criterion = ContrastiveLoss(margin=1.2)
    optimizer = torch.optim.AdamW(model_refiner.parameters(), lr=0.002, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs//3)
    device = next(model_refiner.parameters()).device
    
    if best_params is None:
        best_params = {
            "weights": 0.4088, "context": True, "context_len": 2, "preprocess": False, 
            "norm": 'max', "n_neighbors": 6, "n_components": 4, "umap_metric": 'correlation', 
            "cluster_metric": 'euclidean', "algorithm": 'generic', "cluster_selection_method": 'leaf'
        }
    
    best_loss = float('inf')
    best_reward = -float('inf')
    patience = 15
    no_improve = 0
    
    for epoch in range(epochs):
        total_loss = 0
        batch_count = 0
        
        random.shuffle(data)
        
        for sample in data[:min(len(data), 40)]:
            if len(sample['sentences']) < 3 or len(set(sample['clusters'])) < 2:
                continue
                
            sentences = sample['sentences']
            true_clusters = sample['clusters']
            
            embeddings = model.encode(sentences, show_progress_bar=False)
            embedding_tensor = torch.tensor(embeddings, dtype=torch.float32).to(device)
            
            refined_embeddings = model_refiner(embedding_tensor)
            
            loss = criterion(refined_embeddings, true_clusters)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model_refiner.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            batch_count += 1
        
        scheduler.step()
        
        if batch_count > 0:
            avg_loss = total_loss / batch_count
            avg_reward = evaluate_model_on_validation(model_refiner, val_data, best_params, attention=True)
            
            if epoch % 5 == 0:
                print(f"Epoch {epoch}: Loss: {avg_loss:.4f}, Reward: {avg_reward:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}")
            
            if avg_reward > best_reward:
                best_reward = avg_reward
                best_loss = avg_loss
                no_improve = 0
                
                checkpoint = {
                    'model_state_dict': model_refiner.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'epoch': epoch + 1,
                    'best_score': best_reward,
                    'best_params': best_params,
                    'training_method': 'contrastive'
                }
                save_best_model(checkpoint)
                if epoch % 5 == 0:
                    print(f"  New best model saved! Score: {best_reward:.4f}")
            else:
                no_improve += 1
                if no_improve >= patience:
                    print(f"Early stopping at epoch {epoch}")
                    break
    
    return best_reward

def train_triplet(model_refiner, data, val_data, epochs=100, best_params=None):
    logger.info("Training with triplets...")
    model = load_sent_transformer()
    criterion = TripletLoss(margin=0.8)
    optimizer = torch.optim.AdamW(model_refiner.parameters(), lr=0.002, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs//3)
    device = next(model_refiner.parameters()).device
    
    if best_params is None:
        best_params = {
            "weights": 0.4088, "context": True, "context_len": 2, "preprocess": False, 
            "norm": 'max', "n_neighbors": 6, "n_components": 4, "umap_metric": 'correlation', 
            "cluster_metric": 'euclidean', "algorithm": 'generic', "cluster_selection_method": 'leaf'
        }
    
    best_loss = float('inf')
    best_reward = -float('inf')
    patience = 15
    no_improve = 0
    
    for epoch in range(epochs):
        total_loss = 0
        batch_count = 0
        
        random.shuffle(data)
        
        for sample in data[:min(len(data), 35)]:
            if len(sample['sentences']) < 4 or len(set(sample['clusters'])) < 2:
                continue
                
            sentences = sample['sentences']
            true_clusters = sample['clusters']
            
            triplets = create_triplets(sample, num_triplets=50)
            if not triplets:
                continue
            
            embeddings = model.encode(sentences, show_progress_bar=False)
            embedding_tensor = torch.tensor(embeddings, dtype=torch.float32).to(device)
            
            refined_embeddings = model_refiner(embedding_tensor)
            
            anchor_embeddings = refined_embeddings[[t[0] for t in triplets]]
            positive_embeddings = refined_embeddings[[t[1] for t in triplets]]
            negative_embeddings = refined_embeddings[[t[2] for t in triplets]]
            
            loss = criterion(anchor_embeddings, positive_embeddings, negative_embeddings)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model_refiner.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            batch_count += 1
        
        scheduler.step()
        
        if batch_count > 0:
            avg_loss = total_loss / batch_count
            avg_reward = evaluate_model_on_validation(model_refiner, val_data, best_params, attention=True)
            
            if epoch % 5 == 0:
                print(f"Epoch {epoch}: Loss: {avg_loss:.4f}, Reward: {avg_reward:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}")
            
            if avg_reward > best_reward:
                best_reward = avg_reward
                best_loss = avg_loss
                no_improve = 0
                
                checkpoint = {
                    'model_state_dict': model_refiner.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'epoch': epoch + 1,
                    'best_score': best_reward,
                    'best_params': best_params,
                    'training_method': 'triplet'
                }
                save_best_model(checkpoint)
                if epoch % 5 == 0:
                    print(f"  New best model saved! Score: {best_reward:.4f}")
            else:
                no_improve += 1
                if no_improve >= patience:
                    print(f"Early stopping at epoch {epoch}")
                    break
    
    return best_reward

def hybrid_training(model_refiner, data, val_data, es_generations=40, diff_epochs=50, best_params=None):
    logger.info("Hybrid training...")
    if best_params is None:
        best_params = {
            "weights": 0.4088, "context": True, "context_len": 2, "preprocess": False, 
            "norm": 'max', "n_neighbors": 6, "n_components": 4, "umap_metric": 'correlation', 
            "cluster_metric": 'euclidean', "algorithm": 'generic', "cluster_selection_method": 'leaf'
        }
    
    print("Phase 1: Evolutionary Strategy")
    model_refiner = evolutionary_strategy_training(
        model_refiner, data, 
        generations=es_generations, 
        population_size=50,
        mutation_std=0.12
    )
    
    es_score = evaluate_model_on_validation(model_refiner, val_data, best_params, attention=True)
    print(f"ES final score: {es_score:.4f}")
    
    es_state = {name: param.clone() for name, param in model_refiner.named_parameters()}
    
    print("Phase 2: Differentiable Fine-tuning")
    diff_score = train_with_differentiable_loss(model_refiner, data, val_data, epochs=diff_epochs, best_params=best_params)
    
    print(f"Differentiable score: {diff_score:.4f}")
    
    if diff_score < es_score - 0.01:
        print(f"Differentiable training degraded performance, reverting to ES solution")
        for name, param in model_refiner.named_parameters():
            param.data.copy_(es_state[name])
        final_score = es_score
    else:
        diff_state = {name: param.clone() for name, param in model_refiner.named_parameters()}
        
        print("Phase 3: Triplet Refinement")
        final_score = train_triplet(model_refiner, data[:min(len(data), 25)], val_data, epochs=20, best_params=best_params)
        
        print(f"Final score after triplet: {final_score:.4f}")
        
        best_score = max(es_score, diff_score, final_score)
        if best_score == es_score:
            print("ES was best, using ES solution")
            for name, param in model_refiner.named_parameters():
                param.data.copy_(es_state[name])
            final_score = es_score
        elif best_score == diff_score:
            print("Differentiable was best, using differentiable solution")
            for name, param in model_refiner.named_parameters():
                param.data.copy_(diff_state[name])
            final_score = diff_score
        else:
            print("Triplet was best, keeping current solution")
    
    checkpoint = {
        'model_state_dict': model_refiner.state_dict(),
        'epoch': es_generations + diff_epochs + 20,
        'best_score': final_score,
        'best_params': best_params,
        'training_method': 'hybrid'
    }
    save_best_model(checkpoint)
    
    return final_score

def train_meta_learning(model_refiner, data, val_data, epochs=100, meta_lr=1e-4, task_lr=1e-2, 
                       tasks_per_epoch=20, inner_steps=5, best_params=None):
    logger.info("Training with meta learning...")
    model = load_sent_transformer()
    device = next(model_refiner.parameters()).device
    meta_optimizer = torch.optim.AdamW(model_refiner.parameters(), lr=meta_lr, weight_decay=1e-5)
    criterion = TripletLoss(margin=0.5).to(device)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(meta_optimizer, T_0=20, T_mult=2)
    
    best_val_score = -float('inf')
    patience = 40
    no_improve = 0
    
    if best_params is None:
        best_params = {
            "weights": 0.4088, "context": True, "context_len": 2, "preprocess": False, 
            "norm": 'max', "n_neighbors": 6, "n_components": 4, "umap_metric": 'correlation', 
            "cluster_metric": 'euclidean', "algorithm": 'generic', "cluster_selection_method": 'leaf'
        }

    data_pool = data * 3
    random.shuffle(data_pool)

    for epoch in tqdm(range(epochs), desc="Meta-learning"):
        model_refiner.train()
        epoch_meta_loss = 0
        epoch_valid_tasks = 0
        
        batch_tasks = min(4, tasks_per_epoch // 4) if epoch < 20 else tasks_per_epoch
        
        for batch in range(4):
            meta_optimizer.zero_grad()
            batch_meta_loss = 0
            batch_valid_tasks = 0
            
            for task in range(batch_tasks):
                task_size = random.randint(6, 12)
                task_samples = random.sample(data_pool, min(task_size, len(data_pool)))
                if len(task_samples) < 4:
                    continue
                
                split_point = len(task_samples) // 2
                support_samples = task_samples[:split_point]
                query_samples = task_samples[split_point:]
                
                task_model = copy.deepcopy(model_refiner)
                task_optimizer = torch.optim.SGD(task_model.parameters(), lr=task_lr)
                
                for inner_step in range(inner_steps):
                    inner_loss = 0
                    inner_count = 0
                    
                    for sample in support_samples:
                        if len(sample['sentences']) < 4 or len(set(sample['clusters'])) < 2:
                            continue
                        
                        triplets = create_triplets(sample, num_triplets=40)
                        if not triplets:
                            continue
                        
                        base_embeddings = model.encode(sample['sentences'], show_progress_bar=False)
                        base_embeddings_tensor = torch.tensor(base_embeddings, dtype=torch.float32).to(device)
                        
                        refined_embeddings = task_model(base_embeddings_tensor)
                        
                        anchor_embeddings = refined_embeddings[[t[0] for t in triplets]]
                        positive_embeddings = refined_embeddings[[t[1] for t in triplets]]
                        negative_embeddings = refined_embeddings[[t[2] for t in triplets]]
                        
                        loss = criterion(anchor_embeddings, positive_embeddings, negative_embeddings)
                        inner_loss += loss
                        inner_count += 1
                    
                    if inner_count > 0:
                        avg_inner_loss = inner_loss / inner_count
                        task_optimizer.zero_grad()
                        avg_inner_loss.backward(retain_graph=True)
                        torch.nn.utils.clip_grad_norm_(task_model.parameters(), max_norm=0.5)
                        task_optimizer.step()
                
                query_loss = 0
                query_count = 0
                
                for sample in query_samples:
                    if len(sample['sentences']) < 4 or len(set(sample['clusters'])) < 2:
                        continue
                    
                    triplets = create_triplets(sample, num_triplets=30)
                    if not triplets:
                        continue
                    
                    base_embeddings = model.encode(sample['sentences'], show_progress_bar=False)
                    base_embeddings_tensor = torch.tensor(base_embeddings, dtype=torch.float32).to(device)
                    
                    refined_embeddings = task_model(base_embeddings_tensor)
                    
                    anchor_embeddings = refined_embeddings[[t[0] for t in triplets]]
                    positive_embeddings = refined_embeddings[[t[1] for t in triplets]]
                    negative_embeddings = refined_embeddings[[t[2] for t in triplets]]
                    
                    loss = criterion(anchor_embeddings, positive_embeddings, negative_embeddings)
                    query_loss += loss
                    query_count += 1
                
                if query_count > 0:
                    avg_query_loss = query_loss / query_count
                    
                    grads = torch.autograd.grad(avg_query_loss, task_model.parameters(), 
                                              retain_graph=False, create_graph=False, allow_unused=True)
                    
                    for param, grad in zip(model_refiner.parameters(), grads):
                        if grad is not None:
                            if param.grad is None:
                                param.grad = torch.zeros_like(param.data)
                            param.grad.data.add_(grad.data / (batch_tasks * 4))
                    
                    batch_meta_loss += avg_query_loss.item()
                    batch_valid_tasks += 1
            
            if batch_valid_tasks > 0:
                torch.nn.utils.clip_grad_norm_(model_refiner.parameters(), max_norm=0.5)
                meta_optimizer.step()
                epoch_meta_loss += batch_meta_loss
                epoch_valid_tasks += batch_valid_tasks
        
        scheduler.step()
        
        if epoch_valid_tasks > 0:
            avg_meta_loss = epoch_meta_loss / epoch_valid_tasks
            
            model_refiner.eval()
            val_reward = 0
            val_count = 0
            
            with torch.no_grad():
                for sample in val_data:
                    try:
                        pred_clusters = cluster_sentences(
                            sentences=sample['sentences'], 
                            att_model=model_refiner, 
                            attention=True, 
                            **best_params
                        )
                        if len(pred_clusters) == len(sample['clusters']):
                            val_reward += reward_function(pred_clusters, sample['clusters'])
                            val_count += 1
                    except:
                        continue
            
            avg_val_reward = val_reward / val_count if val_count > 0 else 0
            
            print(f"Epoch {epoch+1}/{epochs}")
            print(f"  Meta Loss: {avg_meta_loss:.4f}")
            print(f"  Val Reward: {avg_val_reward:.4f}")
            print(f"  LR: {scheduler.get_last_lr()[0]:.6f}")
            
            if avg_val_reward > best_val_score:
                best_val_score = avg_val_reward
                no_improve = 0
                
                checkpoint = {
                    'model_state_dict': model_refiner.state_dict(),
                    'optimizer_state_dict': meta_optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'epoch': epoch + 1,
                    'best_score': best_val_score,
                    'best_params': best_params,
                    'training_method': 'meta'
                }
                
                save_best_model(checkpoint)
                print(f"  New best meta-learning model saved! Score: {best_val_score:.4f}")
            else:
                no_improve += 1
                if no_improve >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
    
    return best_val_score

def save_best_model(checkpoint, model_path="models/best_attention_model.pth"):
    logger.info("Saving best model...")
    if not os.path.exists("models"):
        os.makedirs("models")
    
    required_fields = ['model_state_dict', 'epoch', 'best_score', 'training_method']
    for field in required_fields:
        if field not in checkpoint:
            if field == 'training_method':
                checkpoint[field] = 'unknown'
            elif field == 'best_score':
                checkpoint[field] = 0.0
            elif field == 'epoch':
                checkpoint[field] = 0
    
    torch.save(checkpoint, model_path)

def evaluate_model_on_validation(model_refiner, val_data, best_params, attention=True):
    logger.info("Evaluating model on validation...")
    model_refiner.eval()
    val_reward = 0
    val_count = 0
    
    with torch.no_grad():
        for sample in val_data:
            try:
                pred_clusters = cluster_sentences(
                    sentences=sample['sentences'], 
                    att_model=model_refiner if attention else None, 
                    attention=attention, 
                    **best_params
                )
                if len(pred_clusters) == len(sample['clusters']):
                    val_reward += reward_function(pred_clusters, sample['clusters'])
                    val_count += 1
            except:
                continue
    
    return val_reward / val_count if val_count > 0 else 0.0

def train_self_attention_model(epochs=2000, lr=1e-4, resume_from_best=True, training_method='triplet'):
    logger.info(f"Starting Self-Attention model training with {training_method} method...")
    model = load_sent_transformer()
    
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS device for training.")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA device for training.")
    else:
        device = torch.device("cpu")
        print("Using CPU for training.")

    EMBED_DIM = 384
    model_refiner = SelfAttentionModel(embed_dim=EMBED_DIM).to(device)
    
    start_epoch = 0
    best_val_score = 0
    
    if resume_from_best and os.path.exists("models/best_attention_model.pth"):
        try:
            checkpoint = torch.load("models/best_attention_model.pth", map_location=device)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model_refiner.load_state_dict(checkpoint['model_state_dict'])
                start_epoch = checkpoint.get('epoch', 0)
                best_val_score = checkpoint.get('best_score', 0)
                print(f"Resumed from epoch {start_epoch} with best score {best_val_score:.4f}")
            else:
                model_refiner.load_state_dict(checkpoint)
                print("Loaded model weights from best_attention_model.pth")
        except Exception as e:
            print(f"Failed to load checkpoint: {e}")
            print("Starting from scratch...")
    
    samples = load_samples()
    random.seed(42)
    random.shuffle(samples)
    
    split_idx = int(0.8 * len(samples))
    train_samples = samples[:split_idx]
    val_samples = samples[split_idx:]
    
    best_params = {
        "weights": 0.4088, "context": True, "context_len": 2, "preprocess": False, 
        "norm": 'max', "n_neighbors": 6, "n_components": 4, "umap_metric": 'correlation', 
        "cluster_metric": 'euclidean', "algorithm": 'generic', "cluster_selection_method": 'leaf'
    }
    
    if training_method == 'triplet':
        optimizer = torch.optim.AdamW(model_refiner.parameters(), lr=lr, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)
        criterion = TripletLoss(margin=0.5).to(device)
        
        patience = 200
        no_improve_count = 0
        
        for epoch in tqdm(range(start_epoch, epochs)):
            model_refiner.train()
            total_epoch_loss = 0
            batch_count = 0
            
            random.shuffle(train_samples)
            
            for sample in train_samples:
                if len(sample['sentences']) < 4 or len(set(sample['clusters'])) < 2:
                    continue

                triplets = create_triplets(sample)
                if not triplets:
                    continue

                base_embeddings = model.encode(sample['sentences'], show_progress_bar=False)
                base_embeddings_tensor = torch.tensor(base_embeddings, dtype=torch.float32).to(device)
                
                refined_embeddings = model_refiner(base_embeddings_tensor)

                anchor_embeddings = refined_embeddings[[t[0] for t in triplets]]
                positive_embeddings = refined_embeddings[[t[1] for t in triplets]]
                negative_embeddings = refined_embeddings[[t[2] for t in triplets]]
                
                loss = criterion(anchor_embeddings, positive_embeddings, negative_embeddings)
                
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model_refiner.parameters(), max_norm=1.0)
                optimizer.step()
                
                total_epoch_loss += loss.item()
                batch_count += 1
            
            scheduler.step()
            avg_loss = total_epoch_loss / batch_count if batch_count > 0 else 0
            
            model_refiner.eval()
            val_reward = 0
            val_count = 0
            
            with torch.no_grad():
                for sample in val_samples:
                    try:
                        pred_clusters = cluster_sentences(
                            sentences=sample['sentences'], 
                            att_model=model_refiner, 
                            attention=True, 
                            **best_params
                        )
                        if len(pred_clusters) == len(sample['clusters']):
                            val_reward += reward_function(pred_clusters, sample['clusters'])
                            val_count += 1
                    except:
                        continue
            
            avg_val_reward = val_reward / val_count if val_count > 0 else 0
            
            if (epoch + 1) % 5 == 0:
                print(f"Epoch {epoch+1}/{epochs}")
                print(f"  Train Loss: {avg_loss:.4f}")
                print(f"  Val Reward: {avg_val_reward:.4f}")
                print(f"  LR: {scheduler.get_last_lr()[0]:.6f}")
            
            if avg_val_reward > best_val_score:
                best_val_score = avg_val_reward
                no_improve_count = 0
                
                checkpoint = {
                    'model_state_dict': model_refiner.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'epoch': epoch + 1,
                    'best_score': best_val_score,
                    'best_params': best_params,
                    'training_method': 'triplet'
                }
                
                save_best_model(checkpoint)
                if (epoch + 1) % 5 == 0:
                    print(f"  New best model saved! Score: {best_val_score:.4f}")
            else:
                no_improve_count += 1
                if no_improve_count >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
    
    elif training_method == 'contrastive':
        best_val_score = train_contrastive(model_refiner, train_samples, val_samples, epochs, best_params)
    elif training_method == 'differentiable':
        best_val_score = train_with_differentiable_loss(model_refiner, train_samples, val_samples, epochs, best_params)
    elif training_method == 'policy_gradient':
        train_with_policy_gradient(model_refiner, train_samples, epochs)
        best_val_score = evaluate_model_on_validation(model_refiner, val_samples, best_params, attention=True)
        checkpoint = {
            'model_state_dict': model_refiner.state_dict(),
            'epoch': epochs,
            'best_score': best_val_score,
            'best_params': best_params,
            'training_method': 'policy_gradient'
        }
        save_best_model(checkpoint)
    elif training_method == 'evolutionary':
        evolutionary_strategy_training(model_refiner, train_samples, generations=epochs//15, population_size=40)
        best_val_score = evaluate_model_on_validation(model_refiner, val_samples, best_params, attention=True)
        checkpoint = {
            'model_state_dict': model_refiner.state_dict(),
            'epoch': epochs,
            'best_score': best_val_score,
            'best_params': best_params,
            'training_method': 'evolutionary'
        }
        save_best_model(checkpoint)
    elif training_method == 'hybrid':
        best_val_score = hybrid_training(model_refiner, train_samples, val_samples, es_generations=epochs//40, diff_epochs=epochs//20, best_params=best_params)
    elif training_method == 'meta':
        best_val_score = train_meta_learning(model_refiner, train_samples, val_samples, epochs=epochs, meta_lr=lr, best_params=best_params)

    if training_method in ['contrastive', 'differentiable', 'triplet', 'hybrid', 'meta']:
        final_val_score = best_val_score
    else:
        final_val_score = evaluate_model_on_validation(model_refiner, val_samples, best_params, attention=True)
        
        if final_val_score > best_val_score:
            best_val_score = final_val_score
            
            checkpoint = {
                'model_state_dict': model_refiner.state_dict(),
                'epoch': epochs,
                'best_score': best_val_score,
                'best_params': best_params,
                'training_method': training_method
            }
            
            save_best_model(checkpoint)
            print(f"Best model saved! Score: {best_val_score:.4f}")

    print("Training finished.")
    
    final_checkpoint = {
        'model_state_dict': model_refiner.state_dict(),
        'epoch': epochs,
        'final_score': final_val_score,
        'best_params': best_params,
        'training_method': training_method
    }
    
    if not os.path.exists("models"):
        os.makedirs("models")
    torch.save(final_checkpoint, "models/final_attention_model.pth")
    print(f"Final model saved. Best validation score: {best_val_score:.4f}")


def load_aa_rep_sets(dir_path: str = 'data/aa'):
    import re
    sets = []
    if not os.path.isdir(dir_path):
        return sets
    try:
        entries = sorted(os.listdir(dir_path))
    except Exception:
        return sets
    for name in entries:
        folder = os.path.join(dir_path, name)
        if not os.path.isdir(folder):
            continue
        try:
            rep_files = [f for f in os.listdir(folder) if f.lower().startswith('reps') and f.lower().endswith('.txt')]
            if not rep_files:
                rep_files = [f for f in os.listdir(folder) if f.lower().endswith('.txt')]
            if not rep_files:
                continue
            rep_files.sort(key=lambda fn: os.path.getmtime(os.path.join(folder, fn)), reverse=True)
            rep_path = os.path.join(folder, rep_files[0])
            with open(rep_path, 'r') as f:
                content = f.read()
            matches = re.findall(r"SentH\[text=(.*?)\]", content, flags=re.DOTALL)
            if not matches:
                continue
            rep_texts = [m.strip() for i, m in enumerate(matches) if i % 2 == 1 and m.strip()]
            texts = [SentenceHolder(text=t) for t in rep_texts]
            if texts:
                sets.append(texts)
        except Exception:
            continue
    return sets

def _score_rep_clusters(rep_sentences, clusters):
    """Score clustered representative sentences.
    Improvements:
    - Penalize noise (unassigned sentences) proportionally.
    - Use Shannon entropy for source diversity within clusters.
    - Use standard deviation of dates for temporal coherence (lower std => higher coherence).
    Returns a score roughly in [-1, 1]. Higher is better.
    """
    from sklearn.metrics import silhouette_score
    from sklearn.preprocessing import normalize as sk_normalize
    from collections import Counter
    sent_model = load_sent_transformer()

    # Map sentences to indices for label assignment
    key_to_index = {}
    for i, s in enumerate(rep_sentences):
        key_to_index[(s.text, getattr(s, 'source', None), getattr(s, 'date', None))] = i

    n = len(rep_sentences)
    if n == 0:
        return 0.0

    # Assign cluster labels; unassigned remain -1 (noise)
    labels = np.full(n, -1, dtype=int)
    for c in clusters:
        for s in c.get('sentences', []):
            k = (s.text, getattr(s, 'source', None), getattr(s, 'date', None))
            if k in key_to_index:
                labels[key_to_index[k]] = c.get('label', 0)

    assigned_idx = np.where(labels != -1)[0]
    noise_ratio = float(max(0, n - len(assigned_idx)) / n)

    # Silhouette on assigned items only (if valid)
    sil = 0.0
    if len(assigned_idx) >= 3 and len(set(labels[assigned_idx])) >= 2:
        texts = [rep_sentences[i].text for i in assigned_idx]
        embs = sent_model.encode(texts, show_progress_bar=False)
        embs = sk_normalize(embs, norm='l2')
        try:
            sil = float(silhouette_score(embs, labels[assigned_idx]))
        except Exception:
            sil = 0.0

    # Group assigned positions by cluster id
    by_cluster = {}
    for pos, idx in enumerate(assigned_idx):
        lbl = labels[idx]
        by_cluster.setdefault(lbl, []).append(pos)

    # Source diversity via normalized entropy (weighted by cluster size)
    entropy_vals = []
    entropy_weights = []
    for lbl, idxs in by_cluster.items():
        # Sources in this cluster
        sources = [getattr(rep_sentences[assigned_idx[i]], 'source', None) or 'UNK' for i in idxs]
        if not sources:
            continue
        counts = Counter(sources)
        total = float(sum(counts.values()))
        if total <= 0 or len(counts) <= 1:
            entropy_norm = 0.0
        else:
            probs = [c / total for c in counts.values()]
            H = -sum(p * np.log(p + 1e-12) for p in probs)  # natural log
            H_max = np.log(len(counts))
            entropy_norm = float(H / (H_max + 1e-12))
        entropy_vals.append(entropy_norm)
        entropy_weights.append(len(idxs))
    src_entropy = float(np.average(entropy_vals, weights=entropy_weights)) if entropy_vals else 0.0

    # Temporal coherence via std deviation of dates (smaller std => higher coherence)
    # Convert dates to day-scale floats
    from datetime import datetime
    def parse_dt(d):
        if not d:
            return None
        for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d", "%m/%d/%Y"):
            try:
                return datetime.strptime(str(d), fmt)
            except Exception:
                continue
        return None

    tcoh_vals = []
    tcoh_weights = []
    for lbl, idxs in by_cluster.items():
        dts = []
        for i in idxs:
            d = getattr(rep_sentences[assigned_idx[i]], 'date', None)
            dt = parse_dt(d)
            if dt is not None:
                dts.append(dt)
        if len(dts) == 0:
            # Unknown dates: neutral (don’t punish clusters missing metadata)
            continue
        if len(dts) == 1:
            tcoh = 1.0
        else:
            days = np.array([dt.timestamp() / 86400.0 for dt in dts], dtype=float)
            std_days = float(np.std(days))
            # Convert to [0,1]: tighter clusters get values near 1. Scale ~30 days.
            tcoh = float(np.exp(-std_days / 30.0))
        tcoh_vals.append(tcoh)
        tcoh_weights.append(len(idxs))
    temp_coh = float(np.average(tcoh_vals, weights=tcoh_weights)) if tcoh_vals else 1.0

    # Combine metrics with a noise penalty
    noise_weight = 0.3  # penalty strength per fraction of noise
    score = 0.6 * sil + 0.2 * src_entropy + 0.2 * temp_coh - noise_weight * noise_ratio

    # Clamp to reasonable bounds
    return float(max(-1.0, min(1.0, score)))

def objective_group_reps(trial):
    """Optuna objective for grouping representative sentences.
    Uses contextual sentences (second element) and turns off attention/context mixing.
    Also uses a relative min_cluster_size ratio for robustness across set sizes.
    """
    param_preprocess = trial.suggest_categorical("preprocess", [True, False])
    param_norm = trial.suggest_categorical("norm", ['l1', 'l2', 'max', 'none'])
    param_n_neighbors = trial.suggest_int("n_neighbors", 2, 15)
    param_n_components = trial.suggest_int("n_components", 2, 5)
    param_umap_metric = trial.suggest_categorical("umap_metric", ['euclidean', 'manhattan', 'cosine', 'correlation', 'chebyshev'])
    param_cluster_metric = trial.suggest_categorical("cluster_metric", ['euclidean', 'manhattan', 'cosine'])
    param_algorithm = trial.suggest_categorical("algorithm", ['best', 'generic', 'prims_kdtree', 'boruvka_kdtree'])
    param_cluster_selection_method = trial.suggest_categorical("cluster_selection_method", ['eom', 'leaf'])
    param_min_cluster_ratio = trial.suggest_float("min_cluster_ratio", 0.05, 0.3)
    param_min_samples = trial.suggest_int("min_samples", 1, 6)

    base_params = {
        "weights": 1.0,
        "context": False,
        "context_len": 1,
        "preprocess": param_preprocess,
        "attention": False,
        "norm": param_norm,
        "n_neighbors": param_n_neighbors,
        "n_components": param_n_components,
        "umap_metric": param_umap_metric,
        "cluster_metric": param_cluster_metric,
        "algorithm": param_algorithm,
        "cluster_selection_method": param_cluster_selection_method,
        # min_cluster_size computed per set from ratio
        "min_samples": param_min_samples,
    }

    sets = load_aa_rep_sets('data/aa')
    if not sets:
        return 0.0

    total = 0.0
    count = 0
    for rep_set in sets:
        try:
            # Compute absolute min_cluster_size adaptively for this set
            size = max(1, len(rep_set))
            abs_min_cluster_size = max(2, int(round(param_min_cluster_ratio * size)))
            params = dict(base_params)
            params["min_cluster_size"] = abs_min_cluster_size

            clusters = cluster_texts(rep_set, params=params)
            total += _score_rep_clusters(rep_set, clusters)
            count += 1
        except Exception:
            continue
    return total / max(count, 1)

def log_best_trial_to_csv_group_reps(study, trial):
    if trial.state != optuna.trial.TrialState.COMPLETE or study.best_trial.number != trial.number:
        return
    filepath = "optuna_best_trials_log.csv"
    fieldnames = list(trial.params.keys()) + ["value", "objective"]
    file_exists = os.path.isfile(filepath)
    with open(filepath, mode="a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        row = {**trial.params, "value": trial.value, "objective": "group_reps"}
        writer.writerow(row)

def optuna_optimization_group_reps(n_trials: int = 500):
    study = optuna.create_study(direction="maximize")
    try:
        study.optimize(objective_group_reps, n_trials=n_trials, callbacks=[log_best_trial_to_csv_group_reps])
    except KeyboardInterrupt:
        pass
    return study

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(prog='optimize_grouping', description='Optimize clustering and train attention model')
    sub = parser.add_subparsers(dest='cmd')

    p_opt = sub.add_parser('optimize', help='Optimize sentence clustering on labeled samples')
    p_opt.add_argument('--trials', type=int, default=20000)

    p_reps = sub.add_parser('reps', help='Optimize grouping of representative sentences (data/aa)')
    p_reps.add_argument('--trials', type=int, default=500)

    p_train = sub.add_parser('train', help='Train self-attention model')
    p_train.add_argument('--epochs', type=int, default=2000)
    p_train.add_argument('--lr', type=float, default=1e-4)
    p_train.add_argument('--resume', action='store_true')
    p_train.add_argument('--method', type=str, default='triplet', choices=['triplet', 'contrastive', 'differentiable', 'policy_gradient', 'evolutionary', 'hybrid', 'meta'])

    parser.add_argument('--quick', action='store_true', help='Run reps optimization with 100 trials (shortcut)')

    args = parser.parse_args()

    if args.cmd == 'optimize':
        optuna_optimization()
    elif args.cmd == 'reps':
        optuna_optimization_group_reps(n_trials=args.trials)
    elif args.cmd == 'train':
        train_self_attention_model(epochs=args.epochs, lr=args.lr, resume_from_best=args.resume, training_method=args.method)
    else:
        if args.quick:
            optuna_optimization_group_reps(n_trials=100)
        else:
            parser.print_help()