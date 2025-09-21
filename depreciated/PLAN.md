# BIFROST: Complete Implementation Specification
## Bridging Inference Framework for Ordered Structure Transformation

### 1. System Overview

BIFROST generates 3D crystal structures by learning patterns from existing materials databases. It produces atomic arrangements conditioned on desired material properties like electrical conductivity or mechanical strength. The model works by converting crystals into sequences of tokens (like words in a sentence) and generating new sequences that correspond to valid crystal structures.

---

### 2. Background Concepts for Implementation

#### 2.1 Crystal Structure Representation

A crystal structure consists of:
- **Atoms**: Elements from the periodic table (H, Li, Fe, etc.)
- **Unit Cell**: A box that tiles in 3D space (defined by lengths a,b,c and angles α,β,γ)
- **Positions**: Where atoms sit in the unit cell (x,y,z coordinates)
- **Symmetry**: Repeating patterns (one of 230 space groups)

#### 2.2 Wyckoff Positions
Instead of listing every atom position, Wyckoff positions use symmetry to compress information. Example:
- Position "4a" in cubic symmetry means "put atoms at all 4 face centers"
- Some positions have free parameters (x,y,z) that need to be specified
- Total of 1731 Wyckoff positions across all 230 space groups
- We identify 990 unique orbital patterns (some positions appear in multiple space groups)

---

### 3. Data Preprocessing Pipeline

#### 3.1 Structure to Sequence Conversion

**Step 1: Parse Crystal Structure**
```
Input: CIF file or Structure object
Extract:
- Chemical composition: [Li, Fe, P, O] with counts [1, 1, 1, 4]
- Space group number: 62
- Wyckoff positions: Each atom's symmetry position
- Free parameters: x,y,z values for positions that need them
- Lattice parameters: a,b,c,α,β,γ
```

**Step 2: Convert to Token Sequence**
```
Order: [COMPOSITION] [SYMMETRY] [POSITIONS] [LATTICE]

Example for LiFePO4:
[Li] [1] [Fe] [1] [P] [1] [O] [4] [SPACE_62] 
[WYCK_4c] [Li] [x=0.0] [y=0.0] [z=0.0]
[WYCK_4c] [Fe] [x=0.28] [y=0.25] [z=0.97]
[WYCK_4c] [P] [x=0.09] [y=0.25] [z=0.42]
[WYCK_4c] [O] [x=0.10] [y=0.25] [z=0.74]
[WYCK_4c] [O] [x=0.46] [y=0.25] [z=0.20]
[WYCK_8d] [O] [x=0.17] [y=0.04] [z=0.28]
[a=10.3] [b=6.0] [c=4.7] [α=90] [β=90] [γ=90]
```

**Step 3: Create Vocabulary**
- Element tokens: 103 (H through Lr)
- Stoichiometry tokens: 20 (integers 1-20)
- Space group tokens: 230 
- Wyckoff position tokens: 990 unique positions
- Special tokens: [PAD], [EOS], [SEP], [MASK]
- Total discrete vocabulary: ~1350 tokens
- Continuous values: Handled separately (coordinates, lattice parameters)

#### 3.2 Property Data Processing

**Normalize each property to roughly [-1, 1] range:**
```
formation_energy: [-10, 5] eV/atom → divide by 5
bandgap: [0, 12] eV → divide by 6
bulk_modulus: [1, 500] GPa → log scale then divide by 3
density: [0.5, 25] g/cm³ → log scale then divide by 2
elastic_constants: 6x6 matrix → extract eigenvalues, normalize
```

**Handle missing properties:**
- Create binary mask: 1 if property exists, 0 if missing
- For missing values, use placeholder value 0 (will be masked in attention)

---

### 4. Model Architecture Details

#### 4.1 Embedding Layer Implementation

**Discrete Token Embeddings:**
- Create embedding matrices of size [vocab_size, 512]
- Initialize with truncated normal (std=0.02)
- Share embeddings between same token types appearing in different positions

**Continuous Value Encoding:**
For each continuous value v:
1. Apply Gaussian Fourier features at multiple scales:
   ```
   scales = [0.01, 0.1, 1.0, 10.0, 100.0]
   for each scale s:
     features += [sin(v/s), cos(v/s)]
   ```
2. Project to 512 dimensions through MLP:
   ```
   Linear(20) → ReLU → Linear(256) → ReLU → Linear(512)
   ```

**Position Encoding:**
- Maximum sequence length: 250 tokens
- Use standard sinusoidal encoding: PE(pos,2i) = sin(pos/10000^(2i/512))
- Add position encoding to token embeddings

**Token Type Embeddings:**
Add learned embeddings to distinguish:
- ELEMENT tokens (type 0)
- COUNT tokens (type 1)  
- SPACEGROUP tokens (type 2)
- WYCKOFF tokens (type 3)
- COORDINATE tokens (type 4)
- LATTICE tokens (type 5)

#### 4.2 Property Encoder Architecture

**Input Format:**
```
properties_dict = {
    'formation_energy': -2.3,  # eV/atom
    'bandgap': 1.5,           # eV
    'bulk_modulus': None,      # Missing
    'density': 3.2,           # g/cm³
    'target_spacegroup': 225  # Optional constraint
}
mask = [1, 1, 0, 1, 1]  # Binary availability mask
```

**Encoding Process:**
1. **Individual Property Encoding:**
   - Each property gets its own MLP: Linear(1)→SiLU→Linear(64)→SiLU→Linear(512)
   - Missing properties use learned 512-dim parameter vector

2. **Property Relationship Modeling:**
   - Stack encoded properties: [batch_size, num_properties, 512]
   - Apply 2-layer graph neural network where properties are nodes
   - Edge features computed from concatenated node features
   - Message passing updates property representations based on correlations

3. **Output:**
   - Property matrix: [batch_size, num_properties, 512]
   - Presence vector: [batch_size, 512] encoding which properties are available

#### 4.3 Transformer Block with Cross-Attention

**Architecture per block:**

1. **Multi-Head Self-Attention:**
   - 16 attention heads, 32 dims per head
   - Q,K,V projections: Linear(512, 512) each
   - Causal mask for autoregressive generation
   - Attention dropout: 0.1

2. **Cross-Attention to Properties:**
   - Query: Current sequence representation [seq_len, 512]
   - Key/Value: Property embeddings [num_properties, 512]
   - 16 heads with per-head property importance weights
   - Position-specific property attention bias learned during training

3. **Gating Mechanism:**
   ```
   gate_input = Concat([sequence_features, cross_attention_output])
   gate = Sigmoid(Linear(1024, 512))
   output = sequence_features + gate * cross_attention_output
   ```

4. **Feed-Forward Network:**
   ```
   Linear(512, 2048) → GELU → Dropout(0.1) → Linear(2048, 512)
   ```

5. **Layer Normalization:**
   - Applied before each sub-layer (pre-norm)
   - Learned scale and bias parameters

**Total blocks: 16**
- Blocks 0-1: No cross-attention (learn basic patterns)
- Blocks 2-13: Full cross-attention (property conditioning)
- Blocks 14-15: No cross-attention (final refinement)

#### 4.4 Output Head Design

**Dual-Head Architecture:**

1. **Discrete Token Prediction:**
   ```
   Linear(512, 512) → ReLU → LayerNorm → Linear(512, 1350)
   ```
   Output: Logits over vocabulary

2. **Continuous Value Prediction:**
   ```
   Linear(512, 256) → ReLU → Linear(256, 2)
   ```
   Output: (mean, log_variance) for Gaussian sampling

**Token Type Router:**
- Predicts whether next token is discrete or continuous
- Binary classifier: Linear(512, 2)
- Used to select which head to use

---

### 5. Training Procedure

#### 5.1 Dataset Preparation

**Required data:**
- Crystal structures from Materials Project (~150k) and Alexandria (~2M)
- At least one property per structure (typically formation energy from DFT)
- Filter out structures with >15 Wyckoff sites (computational limit)
- Remove space group 1 (no symmetry)

**Data splits:**
- Training: 90%
- Validation: 5%
- Test: 5%
- Ensure chemical diversity in validation/test

#### 5.2 Training Loop

**Batch preparation:**
1. Sample batch_size=256 structures
2. Convert to token sequences (pad to max_length=250)
3. For each structure, randomly mask 30% of available properties
4. Create attention masks for padding

**Forward pass:**
1. Embed token sequences
2. Encode properties (including missing ones)
3. Pass through transformer blocks with cross-attention
4. Compute predictions for next token

**Loss computation:**
```
total_loss = 0
if token_is_discrete:
    total_loss += CrossEntropy(predicted_logits, target_token)
else:
    total_loss += GaussianNLL(predicted_mean, predicted_var, target_value)

# Auxiliary losses
total_loss += 0.1 * property_prediction_loss  # Predict missing properties
total_loss += 0.01 * attention_diversity_loss  # Encourage diverse attention
```

**Optimization:**
- AdamW optimizer with weight decay 0.01
- Learning rate schedule: Linear warmup to 2e-4 over 10k steps, then cosine decay
- Gradient clipping at norm 1.0
- Mixed precision training (FP16) with dynamic loss scaling

#### 5.3 Curriculum Learning Schedule

**Epochs 0-50:** Simple structures only
- Max 5 unique elements
- High symmetry space groups (cubic, tetragonal)
- Max 5 Wyckoff sites

**Epochs 51-150:** Medium complexity
- Max 10 unique elements
- All space groups
- Max 10 Wyckoff sites

**Epochs 151+:** Full complexity
- All structures in dataset
- Up to 15 Wyckoff sites

---

### 6. Inference Pipeline

#### 6.1 Generation Process

**Input:** Property requirements
```
requested_properties = {
    'bandgap': 1.5,  # Want a semiconductor
    'density': 3.0,  # Lightweight
}
```

**Step-by-step generation:**

1. **Initialize:**
   - Encode requested properties (fill missing with learned embeddings)
   - Start sequence with [START] token
   - Set temperature: 0.8 for discrete, 0.6 for continuous

2. **Autoregressive generation loop:**
   ```
   sequence = [START]
   while not done:
       # Get model predictions
       embeddings = embed(sequence)
       hidden = transformer(embeddings, property_embeddings)
       
       # Predict token type
       is_discrete = predict_token_type(hidden[-1])
       
       if is_discrete:
           logits = discrete_head(hidden[-1])
           # Apply temperature
           logits = logits / temperature_discrete
           # Sample from categorical distribution
           next_token = sample_categorical(softmax(logits))
           
           # Apply constraints (e.g., charge neutrality)
           if violates_constraints(next_token, sequence):
               # Mask and resample
               logits[next_token] = -inf
               next_token = sample_categorical(softmax(logits))
       else:
           mean, log_var = continuous_head(hidden[-1])
           # Sample from Gaussian
           std = exp(0.5 * log_var) * temperature_continuous
           value = mean + std * randn()
           # Clip to valid range
           value = clip(value, min_val, max_val)
           next_token = value
       
       sequence.append(next_token)
       
       if next_token == EOS or len(sequence) > 250:
           done = True
   ```

3. **Post-process:**
   - Parse sequence back to crystal structure
   - Validate symmetry consistency
   - Check for overlapping atoms (distance < 0.7 Å)
   - Ensure charge neutrality

#### 6.2 Beam Search for Discrete Tokens

For higher quality generation, use beam search (beam_size=5):
1. Maintain top-5 partial sequences
2. For each sequence, generate next token probabilities
3. Keep top-5 overall (sequence_score * next_token_prob)
4. For continuous tokens, use greedy decoding (mean of Gaussian)

#### 6.3 Constraint Enforcement

**Chemical constraints:**
- Oxidation state rules (e.g., O typically -2, Na typically +1)
- Electronegativity ordering
- Maximum coordination numbers

**Implementation:**
```
def check_oxidation_balance(composition):
    total_charge = 0
    for element, count in composition:
        total_charge += common_oxidation_states[element] * count
    return abs(total_charge) < 0.1

def check_minimum_distances(positions, elements):
    for i, j in pairs:
        dist = distance(positions[i], positions[j])
        min_dist = covalent_radii[elements[i]] + covalent_radii[elements[j]]
        if dist < 0.6 * min_dist:
            return False
    return True
```

---

### 7. Evaluation Metrics

#### 7.1 Generation Quality
- **Validity**: Percentage of generated structures that are chemically valid
- **Uniqueness**: Percentage of unique structures (no duplicates)
- **Novelty**: Percentage not in training set
- **Stability**: Distance to convex hull (eV/atom) using M3GNet or ALIGNN

#### 7.2 Property Accuracy
- MAE between requested and achieved properties
- Success rate: Percentage within 10% of target

#### 7.3 Diversity Metrics
- Coverage of chemical space (element combinations)
- Distribution of space groups
- Structural fingerprint diversity (using SOAP or crystal graphs)

---

### 8. Computational Requirements

**Training:**
- GPUs: 8x NVIDIA A100 (40GB) or 4x H100 (80GB)
- RAM: 256GB system memory
- Storage: 500GB for datasets and checkpoints
- Training time: ~72 hours for 2M structures

**Inference:**
- Single GPU with 8GB+ VRAM
- Generation speed: ~100 structures/minute on A100
- Batch generation recommended for efficiency

---

### 9. Key Implementation Challenges and Solutions

**Challenge 1: Mixing discrete and continuous tokens**
- Solution: Use separate heads and token type predictor
- Carefully handle loss masking for each type

**Challenge 2: Wyckoff position consistency**
- Solution: Pre-compute valid position combinations per space group
- Mask invalid positions during generation

**Challenge 3: Property scale differences**
- Solution: Log-transform large-range properties
- Use property-specific normalization
- Learned importance weights in cross-attention

**Challenge 4: Missing property data**
- Solution: Learned missing embeddings
- Auxiliary property prediction task
- Property dropout during training for robustness

---

### 10. Testing and Validation

**Unit tests needed:**
- Wyckoff position parser correctness
- Sequence reconstruction invertibility
- Property encoder with missing values
- Constraint checker accuracy

**Integration tests:**
- Full pipeline: structure → sequence → structure
- Property conditioning influence on generation
- Beam search vs greedy comparison
- Curriculum learning progression

**Validation experiments:**
- Rediscovery test: Can model find known stable materials?
- Interpolation: Generate structures between known materials
- Extrapolation: Generate with extreme property values
- Ablation: Remove cross-attention, compare performance