import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GenerationConfig, PretrainedConfig, PreTrainedModel, GPT2TokenizerFast
import torch
from transformers.modeling_outputs import CausalLMOutput;(torch.cuda.is_available())

 # Should return True if GPU is available
print(torch.cuda.device_count())  # Should return the number of GPUs available
print(torch.cuda.current_device())  # Should return the current device index
print(torch.cuda.get_device_name(torch.cuda.current_device()))

tokenizer = GPT2TokenizerFast.from_pretrained('C:/Users/wonde/Wonder-Griffin/TraXLMistralForCausalLM')

torch.autograd.set_detect_anomaly(True)

# Custom Configuration for TraXL-Mistral
class TraXLMistralConfig(PretrainedConfig):
    model_type = "TraXLMistral"
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = kwargs.get("vocab_size", 50276)
        self.max_len = kwargs.get("max_len", 256)
        self.hidden_size = kwargs.get("hidden_size", 768)
        self.dropout = kwargs.get("dropout", 0.1)
        self.n_layer = kwargs.get("n_layer", 4)
        self.n_head = kwargs.get("n_head", 4)
        self.ff_expansion_factor = kwargs.get("ff_expansion_factor", 4)
        self.rnn_units = kwargs.get("rnn_units", 128)
        self.num_labels = kwargs.get("num_labels", 5)
        self.max_computation_steps = kwargs.get("max_computation_steps", 5)
        self.memory_size = kwargs.get("memory_size", 256)
        self.sparse_attention = kwargs.get("sparse_attention", True)
        self.dynamic_routing = kwargs.get("dynamic_routing", True)
        self.is_decoder = True  # Required for causal LM
        self.tie_word_embeddings = True
        
config = TraXLMistralConfig(
    vocab_size=50276,  # Increase vocabulary size
    max_len=256,  # Increase max sequence length if needed
    hidden_size=768,  # Increase hidden size
    dropout=0.1,
    n_layer=4,  # Increase number of layers
    n_head=4,  # Increase number of attention heads
    ff_expansion_factor=4,  # Increase feedforward network expansion factor
    n_embd=128,
    rnn_units=128,  # Increase RNN units
    num_labels=5,
    max_computation_steps=5,
    memory_size=256,    
)

# Sparse Attention Layer inspired by Mistral
class SparseAttention(nn.Module):
    def __init__(self, hidden_size, n_head):
        super(SparseAttention, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=n_head)

    def forward(self, query, key, value, attn_mask=None):
        attn_output, _ = self.multihead_attn(query, key, value, attn_mask=attn_mask)
        return attn_output

# Adaptive Computation Time Layer with reinforcement learning
class AdaptiveComputationTime(nn.Module):
    def __init__(self, hidden_size, max_computation_steps=3):
        super(AdaptiveComputationTime, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.max_computation_steps = max_computation_steps
        self.halting_projections = nn.Linear(hidden_size, 1)
        self.step_function = nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        batch_size, seq_len, hidden_size = x.size()
        halting_prob = torch.zeros(batch_size, seq_len).to(x.device)
        remainders = torch.zeros(batch_size, seq_len).to(x.device)
        n_updates = torch.zeros(batch_size, seq_len).to(x.device)
        previous_h = torch.zeros_like(x).to(x.device)
        step = 0
        while step < self.max_computation_steps and (halting_prob < 1.0).byte().any():
            p = self.sigmoid(self.halting_projections(x)).squeeze(-1)
            still_running = (halting_prob < 1.0).float()
            new_halted = (halting_prob + p * still_running > 1.0).float() * still_running
            still_running = (halting_prob + p * still_running <= 1.0).float() * still_running
            halting_prob = halting_prob + p * still_running + new_halted
            remainders = remainders + new_halted * (1 - halting_prob)
            n_updates = n_updates + still_running + new_halted
            update_weights = p * still_running + new_halted
            previous_h = ((previous_h * n_updates.unsqueeze(-1) + update_weights.unsqueeze(-1) * x) /
                          (n_updates + 1e-10).unsqueeze(-1))
            x = self.step_function(x)
            step += 1
        return previous_h, remainders, n_updates

# Memory-Augmented Neural Network (MANN)
class MemoryModule(nn.Module):
    def __init__(self, memory_size, hidden_size):
        super(MemoryModule, self).__init__()
        self.memory_size = memory_size  # Define memory_size here
        self.hidden_size = hidden_size  # Define hidden_size
        self.memory = nn.Parameter(torch.randn(memory_size, hidden_size))  # Memory with the correct dimensions
        self.read_head = nn.Linear(hidden_size, memory_size)
        self.write_head = nn.Linear(hidden_size, memory_size)

    def forward(self, x):
        # Ensure x is at least 2D
        if x.dim() == 1:
            x = x.unsqueeze(0)

        batch_size, seq_len, hidden_size = x.size()  # Expect x to have shape (batch_size, seq_len, hidden_size)
        
        # Compute read weights and read from memory
        read_weights = F.softmax(self.read_head(x), dim=-1)  # Shape: (batch_size, seq_len, memory_size)
        read_memory = torch.matmul(read_weights, self.memory)  # Reading from memory with shape (batch_size, seq_len, hidden_size)

        # Compute the memory update
        write_weights = F.softmax(self.write_head(x), dim=-1)  # Shape: (batch_size, seq_len, memory_size)

        # Ensure correct reshaping of weights and x
        write_weights = write_weights.view(batch_size * seq_len, self.memory_size, 1)  # Shape: (batch_size * seq_len, memory_size, 1)
        x = x.view(batch_size * seq_len, 1, hidden_size)  # Shape: (batch_size * seq_len, 1, hidden_size)

        # Calculate memory update using batch matrix multiplication
        memory_update = torch.bmm(write_weights, x)  # Shape: (batch_size * seq_len, memory_size, hidden_size)

        # Sum over the batch and sequence length dimensions to reduce to memory_size and hidden_size
        memory_update = memory_update.sum(dim=0)  # Shape: (memory_size, hidden_size)

        # Check that the shapes match
        if memory_update.shape != self.memory.shape:
            raise RuntimeError(f"Memory size mismatch: self.memory.shape={self.memory.shape}, memory_update.shape={memory_update.shape}")

        # Safely update the memory without affecting gradient tracking
        with torch.no_grad():
            self.memory.data += memory_update

        return read_memory

# Logical Transformer Layer for reasoning
class LogicalTransformerLayer(nn.Module):
    def __init__(self, hidden_size):
        super(LogicalTransformerLayer, self).__init__()
        self.logic_layer = nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        logic_out = F.relu(self.logic_layer(x))
        return logic_out

class DNCModule(nn.Module):
    def __init__(self, input_size, hidden_size, memory_size=128, memory_dim=8, num_layers=1):
        super(DNCModule, self).__init__()
        self.dnc = nn.ModuleList([nn.LSTM(input_size, hidden_size) for _ in range(num_layers)])
        self.memory = MemoryModule(memory_size, memory_dim)

    def forward(self, x, mem):
        output = x
        for lstm in self.dnc:
            output, (hidden, cell) = lstm(output)
            output = self.memory(output) + output
        return output
    
# Latent Space Clustering
class LatentSpaceClustering(nn.Module):
    def __init__(self, hidden_size, num_clusters):
        super(LatentSpaceClustering, self).__init__()
        self.cluster_centers = nn.Parameter(torch.randn(num_clusters, hidden_size))

    def forward(self, x):
        x_expanded = x.unsqueeze(1)
        distances = torch.cdist(x_expanded, self.cluster_centers)
        cluster_assignments = distances.argmin(dim=-1)
        return cluster_assignments

# Define the custom model TraXL-Mistral
class TraXLMistralForCausalLM(PreTrainedModel):
    config_class = TraXLMistralConfig

    def __init__(self, config, task_type="causal_lm"):
        super(TraXLMistralForCausalLM, self).__init__(config)
        self.task_type = task_type

        # Model layers (embedding, LSTM, transformer, etc.)
        self.embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.lstm = nn.LSTM(config.hidden_size, config.rnn_units, num_layers=config.n_layer//2, batch_first=True, dropout=config.dropout)
        self.sparse_attention = SparseAttention(config.hidden_size, config.n_head) if config.sparse_attention else None
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=config.rnn_units, nhead=config.n_head, dim_feedforward=config.ff_expansion_factor * config.rnn_units, dropout=config.dropout), 
            num_layers=config.n_layer//2
        )
        self.act = AdaptiveComputationTime(config.rnn_units, config.max_computation_steps)
        self.memory = MemoryModule(config.memory_size, config.rnn_units)
        self.cross_attention = nn.MultiheadAttention(embed_dim=config.rnn_units, num_heads=config.n_head, batch_first=True)
        self.latent_clustering = LatentSpaceClustering(config.rnn_units, num_clusters=10)
        self.logic_layer = LogicalTransformerLayer(config.rnn_units)
        self.lm_head = nn.Linear(config.rnn_units, config.vocab_size)

    def forward(self, input_ids, attention_mask=None, labels=None, past_key_values=None, use_cache=None, **kwargs):
        # Use past_key_values if provided to speed up inference
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]  # Only process the last token

        # Embedding lookup
        x = self.embedding(input_ids)

        # LSTM and Transformer layers
        lstm_out, _ = self.lstm(x)
        transformer_out = self.transformer_encoder(lstm_out)

        # Adaptive computation and memory module
        act_out, remainders, n_updates = self.act(transformer_out)
        memory_out = self.memory(act_out)

        # Cross-attention and logical layers
        attention_out, _ = self.cross_attention(memory_out, memory_out, memory_out)
        logic_out = self.logic_layer(attention_out)

        # Final language model head for causal LM
        logits = self.lm_head(logic_out)

        # Compute loss if labels are provided
        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))

        # Return CausalLMOutput, as expected by the generate function
        return CausalLMOutput(
            loss=loss,
            logits=logits
        )
        # Required for `.generate()` to work
    def prepare_inputs_for_generation(self, input_ids, past=None, **kwargs):
        if past is None:
            # first step in the generation
            return {"input_ids": input_ids}
        else:
            # at every step, we only need to pass the last token for faster generation
            return {"input_ids": input_ids[:, -1:], "past_key_values": past}

    # Optional but recommended for beam search or other advanced generation techniques
    def _reorder_cache(self, past, beam_idx):
        """
        This method helps with the beam search during generation. It reorders the cache to match
        the order of the beam search paths. Implement this if you're using beam search generation.
        """
        return tuple(layer_past.index_select(1, beam_idx) for layer_past in past)
# Example generation config
class RobustGenerationConfig(GenerationConfig):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Default settings (can be dynamically adjusted)
        self.do_sample = kwargs.get('do_sample', True)
        self.max_length = kwargs.get('max_length', 512)
        self.temperature = kwargs.get('temperature', 0.7)
        self.top_p = kwargs.get('top_p', 0.9)
        self.top_k = kwargs.get('top_k', 50)
        self.repetition_penalty = kwargs.get('repetition_penalty', 1.2)
        self.no_repeat_ngram_size = kwargs.get('no_repeat_ngram_size', 3)
        self.length_penalty = kwargs.get('length_penalty', 1.0)
        self.num_beams = kwargs.get('num_beams', 5)
        self.early_stopping = kwargs.get('early_stopping', True)
        self.diversity_penalty = kwargs.get('diversity_penalty', 0.5)  # Added for reducing redundancy
        self.use_cache = kwargs.get('use_cache', True)
        
        # Task-specific defaults (these can be expanded or modified)
        self.task_type = kwargs.get('task_type', 'general')
        if self.task_type == 'summarization':
            self.length_penalty = kwargs.get('length_penalty', 2.0)
            self.max_length = kwargs.get('max_length', 150)
        elif self.task_type == 'conversation':
            self.top_p = kwargs.get('top_p', 0.95)
            self.temperature = kwargs.get('temperature', 0.8)
            self.max_length = kwargs.get('max_length', 50)
        elif self.task_type == 'creative_writing':
            self.top_p = kwargs.get('top_p', 0.9)
            self.temperature = kwargs.get('temperature', 0.9)
            self.max_length = kwargs.get('max_length', 1024)

# Usage Example:
robust_gen_config = RobustGenerationConfig(
    do_sample=False,
    max_length=512,
    temperature=None,
    top_p=None,
    top_k=50,
    repetition_penalty=1.2,
    no_repeat_ngram_size=3,
    length_penalty=1.0,
    num_beams=5,
    early_stopping=True,
    diversity_penalty=1.5,
    task_type='conversation'
)


# Ensure inputs are also on the same device
print(type(input))