"""
Merged ProtoMindBrain v3
- Modular cognitive architecture
- Reward-free, future-focused
- Perception → Memory → Planning → Action
- Arbitration over modules
- Optional ConsultantModule
- Generative model for next observation prediction
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from protomind.core.generative_model import GenerativeModelModule
from protomind.core.memory import AttentionMemory


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# --- Cognitive Modules ---
class PerceptionModule(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        self.feature_extractor = nn.Linear(hidden_size, hidden_size)

    def forward(self, obs):
        h = self.encoder(obs)
        features = self.feature_extractor(h)
        return features, h


class WorkingMemoryModule(nn.Module):
    def __init__(self, hidden_size, memory_slots=16):
        super().__init__()
        self.hidden_size = hidden_size
        self.memory_slots = memory_slots
        self.attention_memory = AttentionMemory(hidden_size, memory_slots, DEVICE)

    def forward(self, x):
        attended, _ = self.attention_memory.attend(x)
        self.attention_memory.update(x)
        return attended


class PlanningModule(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.planner = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        self.plan_head = nn.Linear(hidden_size, hidden_size)

    def forward(self, current_state, memory_context):
        combined = torch.cat([current_state, memory_context], dim=-1)
        planned = self.planner(combined)
        plan_features = self.plan_head(planned)
        return plan_features


class ActionModule(nn.Module):
    def __init__(self, hidden_size, action_size):
        super().__init__()
        self.policy_net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size)
        )

    def forward(self, plan_state):
        logits = self.policy_net(plan_state)
        return logits


# --- Arbitration Network ---
class ArbitrationNet(nn.Module):
    """Outputs soft weights over modules."""
    def __init__(self, module_feat_dim, n_modules, hidden=128):
        super().__init__()
        self.n_modules = n_modules
        self.proj_q = nn.Linear(module_feat_dim, hidden)
        self.proj_k = nn.Linear(module_feat_dim, hidden)
        self.proj_v = nn.Linear(module_feat_dim, hidden)
        self.out = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))
        self.gating_head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, n_modules), nn.Softmax(dim=-1))

    def forward(self, features):
        q = self.proj_q(features)
        k = self.proj_k(features)
        v = self.proj_v(features)
        scores = torch.matmul(q, k.transpose(-2, -1)) / (q.shape[-1] ** 0.5)
        attn = F.softmax(scores, dim=-1)
        fused = torch.matmul(attn, v)
        weights = self.out(fused).squeeze(-1)
        weights = F.softmax(weights, dim=-1)
        gating_weights = self.gating_head(fused.mean(dim=1))
        return weights, fused, gating_weights


# --- ProtoMindBrain ---
class ProtoMindBrain(nn.Module):
    def __init__(self, input_size, hidden_size, action_size, n_modules=3, knowledge_base_path: str = None, use_consultant=False, memory_slots=16):
        super().__init__()
        self.hidden_size = hidden_size
        self.action_size = action_size

        # Instantiate cognitive modules per "module"
        self.module_list = nn.ModuleList([
            nn.ModuleDict({
                'perception': PerceptionModule(input_size, hidden_size),
                'memory': WorkingMemoryModule(hidden_size, memory_slots),
                'planning': PlanningModule(hidden_size),
                'action': ActionModule(hidden_size, action_size)
            }) for _ in range(n_modules)
        ])
        self.n_modules = n_modules

        self.arb = ArbitrationNet(hidden_size, self.n_modules, hidden=hidden_size)
        self.policy_refine = nn.Sequential(nn.Linear(action_size, action_size), nn.Tanh())
        self.generative_model = GenerativeModelModule(input_size, 1, input_size)

    def forward(self, x, temperature: float = 1.0, module_dropout_prob: float = 0.0):
        batch = x.shape[0]
        module_feats = []
        module_logits = []

        # Process each module
        for module in self.module_list:
            p_feat, _ = module['perception'](x)
            m_context = module['memory'](p_feat)
            plan_feat = module['planning'](p_feat, m_context)
            logits = module['action'](plan_feat)
            module_feats.append(plan_feat.unsqueeze(1))
            module_logits.append(logits.unsqueeze(1))



        module_feats = torch.cat(module_feats, dim=1)
        module_logits = torch.cat(module_logits, dim=1)

        # Arbitration
        weights, fused_state, gating_weights = self.arb(module_feats)

        # Temperature scaling
        if temperature != 1.0:
            gating_weights = torch.pow(gating_weights, 1.0 / max(1e-6, temperature))
            gating_weights = gating_weights / gating_weights.sum(dim=-1, keepdim=True)

        # Optional module dropout
        if module_dropout_prob > 0.0 and self.training:
            mask_indices = torch.randint(0, self.n_modules, (batch,), device=x.device)
            drop_mask = (torch.rand(batch, device=x.device) < module_dropout_prob)
            if drop_mask.any():
                gating_weights[drop_mask, mask_indices[drop_mask]] = 0.0
                gating_weights = gating_weights + 1e-8
                gating_weights = gating_weights / gating_weights.sum(dim=-1, keepdim=True)

        # Select module per batch item
        selected_module_indices = torch.multinomial(gating_weights, num_samples=1).squeeze(1)
        combined_logits = module_logits[torch.arange(batch), selected_module_indices]
        refined = combined_logits + self.policy_refine(combined_logits)

        # Generative prediction
        dist = torch.distributions.Categorical(logits=refined)
        action_sample = dist.sample().unsqueeze(-1)
        predicted_next_obs = self.generative_model(x, action_sample.float())

        return refined, weights, fused_state, predicted_next_obs, gating_weights, module_logits, selected_module_indices

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path, device='cpu'):
        self.load_state_dict(torch.load(path, map_location=device))
