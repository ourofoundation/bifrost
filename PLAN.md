# BIFROST: Complete Implementation Specification
## Bridging Inference Framework for Ordered Structure Transformation

### 1. System Overview

BIFROST generates 3D crystal structures conditioned on target material properties. It uses an autoregressive transformer to generate structures as sequences of tokens, with properties specified as discrete bins in a prefix. The model learns from existing materials databases to produce chemically valid, stable structures.

---

### 2. Core Concepts

#### 2.1 Crystal Structure Components
- **Composition**: Elements and their counts (e.g., Li₂FeO₄)
- **Space Group**: One of 230 symmetry groups defining repeating patterns
- **Wyckoff Positions**: Symmetry-unique atomic positions (~990 types)
- **Free Parameters**: x,y,z coordinates for positions requiring them
- **Lattice**: Unit cell dimensions (a,b,c,α,β,γ)

#### 2.2 Sequence Representation
Crystals are converted to linear sequences:
```
[PROPERTY_PREFIX] [SEP] [COMPOSITION] [SYMMETRY] [POSITIONS] [LATTICE] [EOS]
```

---

### 3. Data Pipeline

#### 3.1 Property Discretization

All properties are binned into 4 discrete levels: NONE, LOW, MED, HIGH

```python
property_bins = {
    'energy_above_hull': {
        'thresholds': [0.01, 0.05, 0.1],  # eV/atom
        'tokens': ['EHULL_NONE', 'EHULL_LOW', 'EHULL_MED', 'EHULL_HIGH']
    },
    'band_gap': {
        'thresholds': [0.5, 2.0, 4.0],  # eV
        'tokens': ['BANDGAP_NONE', 'BANDGAP_LOW', 'BANDGAP_MED', 'BANDGAP_HIGH']
    },
    'density': {
        'thresholds': [2.0, 4.0, 8.0],  # g/cm³
        'tokens': ['DENSITY_NONE', 'DENSITY_LOW', 'DENSITY_MED', 'DENSITY_HIGH']
    },
    'bulk_modulus': {
        'thresholds': [50, 150, 300],  # GPa
        'tokens': ['BULK_NONE', 'BULK_LOW', 'BULK_MED', 'BULK_HIGH']
    },
    'formation_energy_per_atom': {
        'thresholds': [-2.0, -0.5, 0.0],  # eV/atom
        'tokens': ['FORM_NONE', 'FORM_LOW', 'FORM_MED', 'FORM_HIGH']
    }
}

def discretize_property(value, thresholds):
    if value < thresholds[0]: return 'NONE'
    elif value < thresholds[1]: return 'LOW'
    elif value < thresholds[2]: return 'MED'
    else: return 'HIGH'
```

#### 3.2 Sequence Construction

**Example sequence for LiFePO₄:**
```
[BANDGAP_MED] [DENSITY_LOW] [MASK] [EHULL_LOW] [SEP]
[Li] [1] [Fe] [1] [P] [1] [O] [4] 
[SPACE_62]
[WYCK_4c] [Li] [x=0.0] [y=0.0] [z=0.0]
[WYCK_4c] [Fe] [x=0.28] [y=0.25] [z=0.97]
[WYCK_4c] [P] [x=0.09] [y=0.25] [z=0.42]
[WYCK_4c] [O] [x=0.10] [y=0.25] [z=0.74]
[a=10.3] [b=6.0] [c=4.7] [α=90] [β=90] [γ=90]
[EOS]
```

#### 3.3 Vocabulary Definition

**Discrete tokens (total ~1430):**
- Elements: 103
- Stoichiometry: 20 (integers 1-20)
- Space groups: 230
- Wyckoff positions: 990
- Property bins: 80 (5 properties × 4 bins × 4 variations)
- Special: [PAD], [EOS], [SEP], [MASK], [UNK]

**Continuous values:**
- Coordinates: x,y,z ∈ [0,1]
- Lattice parameters: a,b,c ∈ [1,100] Å, angles ∈ [30,150]°

---

### 4. Model Architecture

#### 4.1 Embedding Layer

```python
class EmbeddingLayer(nn.Module):
    def __init__(self, vocab_size=1430, d_model=512, max_seq_len=512):
        super().__init__()
        # Discrete token embeddings
        self.token_embed = nn.Embedding(vocab_size, d_model)
        nn.init.normal_(self.token_embed.weight, std=0.02)
        
        # Continuous value encoder
        self.continuous_encoder = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, d_model)
        )
        
        # Positional encoding (sinusoidal)
        self.pos_encoding = self.create_positional_encoding(max_seq_len, d_model)
        
        # Token type embeddings
        self.token_types = nn.Embedding(7, d_model)
        # Types: PROPERTY=0, ELEMENT=1, COUNT=2, SPACEGROUP=3, 
        #        WYCKOFF=4, COORDINATE=5, LATTICE=6
        
    def forward(self, tokens, token_types, is_continuous_mask):
        # Embed discrete tokens
        discrete_embeds = self.token_embed(tokens)
        
        # Handle continuous values
        continuous_embeds = self.continuous_encoder(tokens.unsqueeze(-1))
        
        # Combine based on mask
        embeds = torch.where(
            is_continuous_mask.unsqueeze(-1),
            continuous_embeds,
            discrete_embeds
        )
        
        # Add type and position
        embeds = embeds + self.token_types(token_types)
        embeds = embeds + self.pos_encoding[:tokens.size(1)]
        
        return embeds
```

#### 4.2 Transformer Architecture

```python
class BIFROST(nn.Module):
    def __init__(
        self,
        vocab_size=1430,
        d_model=512,
        n_heads=16,
        n_layers=16,
        d_ff=2048,
        dropout=0.1,
        max_seq_len=512
    ):
        super().__init__()
        self.d_model = d_model
        
        # Embeddings
        self.embeddings = EmbeddingLayer(vocab_size, d_model, max_seq_len)
        
        # Transformer blocks (standard, no cross-attention)
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
        # Output heads
        self.discrete_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.LayerNorm(d_model),
            nn.Linear(d_model, vocab_size)
        )
        
        self.continuous_head = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.ReLU(),
            nn.Linear(256, 2)  # mean, log_var for Gaussian
        )
        
        # Token type predictor
        self.type_predictor = nn.Linear(d_model, 2)  # binary: discrete/continuous
```

#### 4.3 Transformer Block

```python
class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout):
        super().__init__()
        # Multi-head self-attention with causal mask
        self.attention = nn.MultiheadAttention(d_model, n_heads, dropout, batch_first=True)
        self.attn_norm = nn.LayerNorm(d_model)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        self.ffn_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        # Self-attention with residual
        attn_out = self.attention(x, x, x, attn_mask=mask, is_causal=True)[0]
        x = self.attn_norm(x + self.dropout(attn_out))
        
        # FFN with residual
        ffn_out = self.ffn(x)
        x = self.ffn_norm(x + self.dropout(ffn_out))
        
        return x
```

---

### 5. Training

#### 5.1 Dataset Preparation

**Data sources:**
- Materials Project: ~150k structures
- Alexandria (filtered): ~2M structures with Ehull < 0.25 eV/atom
- WBM dataset: ~257k validated structures
- **Total:** ~2.4M structures after filtering

**Processing:**
1. Parse structures to extract composition, symmetry, positions
2. Calculate/retrieve properties (at least formation energy required)
3. Discretize properties into bins
4. Filter structures with >15 Wyckoff sites
5. Remove space group 1 (no symmetry)

**Train/Val/Test split:** 90/5/5 with chemical diversity

#### 5.2 Training Procedure

```python
# Training configuration
config = {
    'batch_size': 256,
    'max_seq_len': 512,
    'learning_rate': 2e-4,
    'warmup_steps': 10000,
    'total_epochs': 300,
    'gradient_clip': 1.0,
    'property_dropout': 0.3,  # Randomly mask 30% of available properties
    'property_removal': 0.1,  # Completely remove 10% of properties
}

def training_step(batch):
    # Prepare sequences with property prefixes
    sequences = []
    for structure in batch:
        # Build property prefix with dropout
        prefix = []
        for prop_name in sorted(structure.available_properties):
            if random.random() > config['property_dropout']:
                if random.random() > config['property_removal']:
                    value_bin = discretize_property(structure[prop_name])
                    prefix.extend([prop_name_token, value_bin_token])
                else:
                    prefix.extend([prop_name_token, MASK_token])
        
        # Add structure tokens
        sequence = prefix + [SEP] + structure.tokens + [EOS]
        sequences.append(sequence)
    
    # Forward pass
    inputs = sequences[:, :-1]
    targets = sequences[:, 1:]
    
    logits = model(inputs)
    
    # Compute loss (CrossEntropy for discrete, Gaussian NLL for continuous)
    discrete_mask = is_discrete_token(targets)
    continuous_mask = ~discrete_mask
    
    loss = CrossEntropyLoss(logits[discrete_mask], targets[discrete_mask])
    loss += GaussianNLLLoss(logits[continuous_mask], targets[continuous_mask])
    
    return loss
```

#### 5.3 Curriculum Learning

**Epochs 0-100:** Simple structures
- Max 5 unique elements
- Max 5 Wyckoff sites  
- High symmetry (cubic, tetragonal)
- Single property conditioning

**Epochs 101-200:** Medium complexity
- Max 10 unique elements
- Max 10 Wyckoff sites
- All space groups
- 2-3 property conditioning

**Epochs 201-300:** Full complexity
- All structures
- Up to 5 property conditioning
- Include rare elements and low-symmetry structures

**Optimization:**
- AdamW with weight decay 0.01
- OneCycleLR scheduler: warmup to 2e-4, cosine decay
- Mixed precision training (FP16)
- Gradient clipping at norm 1.0

---

### 6. Inference

#### 6.1 Generation Process

```python
def generate_structure(property_targets, temperature=0.8, max_length=512):
    """
    property_targets: dict of property_name -> desired_value
    Example: {'band_gap': 2.5, 'density': 3.0}
    """
    # Build property prefix
    prefix_tokens = []
    for prop_name in sorted(property_targets.keys()):
        value = property_targets[prop_name]
        bin_token = discretize_property(value, property_bins[prop_name])
        prefix_tokens.extend([vocab[prop_name], vocab[bin_token]])
    
    # Add separator
    sequence = prefix_tokens + [vocab['SEP']]
    
    # Autoregressive generation
    with torch.no_grad():
        while len(sequence) < max_length:
            # Encode sequence so far
            inputs = torch.tensor(sequence).unsqueeze(0)
            hidden = model.transformer(model.embeddings(inputs))
            
            # Predict next token type
            is_discrete = torch.sigmoid(model.type_predictor(hidden[:, -1])) > 0.5
            
            if is_discrete:
                # Sample discrete token
                logits = model.discrete_head(hidden[:, -1]) / temperature
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, 1).item()
                
                # Apply constraints (charge neutrality, etc.)
                if violates_constraints(next_token, sequence):
                    # Mask and resample
                    logits[next_token] = -float('inf')
                    probs = F.softmax(logits, dim=-1)
                    next_token = torch.multinomial(probs, 1).item()
            else:
                # Sample continuous value
                mean, log_var = model.continuous_head(hidden[:, -1]).split(1, dim=-1)
                std = torch.exp(0.5 * log_var)
                value = mean + std * torch.randn_like(mean) * temperature
                next_token = value.item()
            
            sequence.append(next_token)
            
            if next_token == vocab['EOS']:
                break
    
    return parse_sequence_to_structure(sequence)
```

#### 6.2 Constraint Enforcement

```python
def violates_constraints(token, sequence_so_far):
    # Check charge balance
    if token in element_tokens:
        composition = extract_composition(sequence_so_far + [token])
        if not is_charge_balanced(composition):
            return True
    
    # Check Wyckoff compatibility
    if token in wyckoff_tokens:
        space_group = extract_space_group(sequence_so_far)
        if not is_valid_wyckoff(token, space_group):
            return True
    
    return False
```

#### 6.3 Post-Processing Pipeline

```python
def post_process_structure(structure):
    # 1. Check minimum distances
    if has_overlapping_atoms(structure, min_dist=0.7):  # Angstroms
        return None
    
    # 2. Check connectivity (Laplacian eigenvalues)
    if not is_connected(structure):
        return None
    
    # 3. Relax with M3GNet (fast, approximate)
    structure = relax_m3gnet(structure, steps=50)
    
    # 4. Refine with ORBITAL (slower, more accurate)
    structure = relax_orbital(structure, fmax=0.05)
    
    # 5. Check stability (optional, expensive)
    ehull = calculate_ehull_alignn(structure)
    
    return structure, ehull
```

---

### 7. Evaluation Metrics

#### 7.1 Generation Quality
- **Validity**: % structures passing all constraint checks
- **Uniqueness**: % unique after deduplication
- **Novelty**: % not in training set
- **Stability**: % within 0.1 eV/atom of convex hull

#### 7.2 Property Accuracy
For each target property:
- **MAE**: Mean absolute error vs. DFT-calculated values
- **Success Rate**: % within 20% of target

#### 7.3 Diversity
- **Composition diversity**: Number of unique formulas
- **Structure diversity**: Distribution across space groups
- **Property coverage**: Range of achieved property values

---

### 8. Implementation Details

#### 8.1 Computational Requirements

**Training:**
- Hardware: 4× NVIDIA A100 (40GB) or 2× H100 (80GB)
- Training time: ~48 hours for 2.4M structures
- Memory: 128GB system RAM

**Inference:**
- Single GPU with 8GB+ VRAM
- Generation rate: ~100 structures/minute (including post-processing)

#### 8.2 Key Hyperparameters

```python
hyperparameters = {
    # Model
    'd_model': 512,
    'n_heads': 16,
    'n_layers': 16,
    'd_ff': 2048,
    'dropout': 0.1,
    'vocab_size': 1430,
    
    # Training
    'batch_size': 256,
    'learning_rate': 2e-4,
    'warmup_steps': 10000,
    'weight_decay': 0.01,
    'gradient_clip': 1.0,
    
    # Generation
    'temperature': 0.8,
    'max_length': 512,
    'beam_size': 1,  # Greedy decoding (beam search optional)
    
    # Properties
    'property_dropout': 0.3,
    'n_property_bins': 4,
}
```

#### 8.3 Data Storage Format

```python
# Structure storage format (HDF5 recommended)
structure_data = {
    'structure_id': str,
    'composition': List[Tuple[element, count]],
    'space_group': int,
    'wyckoff_positions': List[Dict],
    'lattice': Dict[str, float],
    'properties': {
        'formation_energy_per_atom': float,
        'band_gap': float,  # Optional
        'density': float,  # Optional
        'bulk_modulus': float,  # Optional
        'energy_above_hull': float,
    },
    'tokens': List[int],  # Pre-tokenized sequence
}
```

---

### 9. Testing Strategy

#### 9.1 Unit Tests
- Tokenization invertibility
- Property discretization correctness
- Constraint checker accuracy
- Wyckoff position validity

#### 9.2 Integration Tests
- Full pipeline: structure → tokens → structure
- Property conditioning influence
- Curriculum learning progression
- Memory leak detection

#### 9.3 Validation Experiments
- **Rediscovery**: Generate known stable materials from MP
- **Interpolation**: Properties between training examples
- **Extrapolation**: Extreme property values
- **Ablation**: Remove property conditioning, measure impact

---

### 10. Common Issues and Solutions

**Issue 1: Mode collapse (generating same structures repeatedly)**
- Solution: Increase temperature, add noise to embeddings

**Issue 2: Poor property targeting**
- Solution: Adjust bin boundaries based on data distribution

**Issue 3: Invalid Wyckoff positions**
- Solution: Pre-compute valid position mask per space group

**Issue 4: Memory overflow with long sequences**
- Solution: Gradient checkpointing, reduce max sequence length

**Issue 5: Slow generation**
- Solution: Batch generation, KV-cache optimization

---

### 11. Code Structure

```
bifrost/
├── data/
│   ├── tokenizer.py         # Structure ↔ token conversion
│   ├── properties.py        # Property binning
│   └── dataset.py           # PyTorch dataset/dataloader
├── model/
│   ├── embeddings.py        # Token and position embeddings
│   ├── transformer.py       # Core transformer blocks
│   ├── bifrost.py          # Main model class
│   └── heads.py            # Output heads
├── training/
│   ├── train.py            # Training loop
│   ├── curriculum.py       # Curriculum scheduling
│   └── optimizer.py        # Learning rate schedules
├── generation/
│   ├── generate.py         # Inference code
│   ├── constraints.py      # Chemical constraints
│   └── postprocess.py      # Relaxation and validation
├── evaluation/
│   ├── metrics.py          # Evaluation metrics
│   └── benchmark.py        # Benchmark suites
└── utils/
    ├── chemistry.py        # Chemical rules and radii
    └── crystallography.py  # Space group and Wyckoff data
```

---

### 12. Summary

BIFROST is a streamlined autoregressive transformer for crystal structure generation with property conditioning. Key design choices:

1. **Discrete property bins** avoid complexity of continuous conditioning
2. **Prefix conditioning** naturally handles missing properties via [MASK] tokens
3. **Standard transformer** without cross-attention keeps architecture simple
4. **Post-generation validation** rather than validity heads reduces model complexity
5. **Curriculum learning** improves training stability and final performance

The model achieves property-guided generation through learned correlations between property prefixes and structural features, without requiring complex architectural additions.