import torch
from torch.fx.experimental.symbolic_shapes import StatefulSymbolicContext
import torch.nn as nn
from torch.nn import functional as F
import numpy as np

# import tiktoken


class SynapseUNet(nn.Module):
    def __init__(self, out_dims, depth, minimum_width=16, dropout=0.0):
        """
        Create a UNet neural network

        Define UNET structure based on depth
        Creates `depth` width values, leading to `depth-1` blocks
        """
        super().__init__()
        self.width_out = out_dims
        self.n_deep = depth  # Store depth just for reference if needed
        widths = np.linspace(out_dims, minimum_width, depth)  # array[768, 16]
        # Initial projection layer
        self.proj1 = nn.Sequential(
            nn.LazyLinear(int(widths[0])),  # Project to the first width
            nn.LayerNorm(int(widths[0])),
            nn.SiLU(),
        )

        # Downward path (encoding layers)
        self.down_proj = nn.ModuleList()
        self.up_proj = nn.ModuleList()
        self.skip_lns = nn.ModuleList()
        num_blocks = len(widths) - 1  # Number of down/up blocks created

        for i in range(num_blocks):
            # Down block: widths[i] -> widths[i+1]
            self.down_proj.append(
                nn.Sequential(
                    nn.Dropout(dropout),
                    nn.Linear(int(widths[i]), int(widths[i + 1])),
                    nn.LayerNorm(int(widths[i + 1])),
                    nn.SiLU(),
                )
            )
            # Up block: widths[i+1] -> widths[i]
            # Note: Up blocks are added in order matching down blocks conceptually,
            # but applied in reverse order in the forward pass.
            self.up_proj.append(
                nn.Sequential(
                    nn.Dropout(dropout),
                    nn.Linear(int(widths[i + 1]), int(widths[i])),
                    nn.LayerNorm(int(widths[i])),
                    nn.SiLU(),
                )
            )
            # Skip connection LayerNorm operates on width[i]
            self.skip_lns.append(nn.LayerNorm(int(widths[i])))

    def forward(self, x):
        # Initial projection
        out_first = self.proj1(x)

        # Downward path, storing outputs for skip connections
        outs_down = [out_first]
        for layer in self.down_proj:
            outs_down.append(layer(outs_down[-1]))
        # outs_down contains [level_0, level_1, ..., level_depth-1=bottleneck] outputs

        # Upward path, starting from the bottleneck output
        outs_up = outs_down[-1]  # Bottleneck activation
        num_blocks = len(self.up_proj)  # Should be depth - 1

        for i in range(num_blocks):
            # Apply up projection in reverse order relative to down blocks
            # up_projection[num_blocks - 1 - i] processes deeper features first
            up_layer_idx = num_blocks - 1 - i
            out_up = self.up_proj[up_layer_idx](outs_up)

            # Get corresponding skip connection from downward path
            # skip_connection index = num_blocks - 1 - i (same as up_layer_idx)
            # This matches the output width of the up_projection[up_layer_idx]
            skip_idx = up_layer_idx
            skip_connection = outs_down[skip_idx]

            # Add skip connection and apply LayerNorm corresponding to this level
            # skip_lns index also corresponds to the level = skip_idx
            outs_up = self.skip_lns[skip_idx](out_up + skip_connection)

        # The final output after all up-projections
        return outs_up


class NLM(nn.Module):
    def __init__(self, in_dims, out_dims, N, T=1.0, dropout=0.1):
        super().__init__()
        """
        SuperLinear, MLP for each dimennsion
        Parameters:
            in_dims=25: "Memory length" length of pre-activation history
            out_dims=4: output dimensions (Hidden dims of memory)
            N=768: num neurons (n_embd)
            T=1: Initial value for learnable temperature/scalingFactor
            dropout=0.1: 
        Dropout-> LayerNorm ->Linear
        self.dropout = nn.Dropout(dropout)
        


        W: (in_dims, out_dims, N)
        b1: (1, N, out_dims)
        """
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(in_dims, elementwise_affine=True)
        self.register_parameter(
            "w1",
            nn.Parameter(
                torch.empty((in_dims, out_dims, N)).uniform_(
                    -((in_dims + out_dims) ** -0.5),
                    (in_dims + out_dims) ** -0.5,
                ),
                requires_grad=True,
            ),
        )
        self.register_parameter(
            "b1", nn.Parameter(torch.zeros((1, N, out_dims)), requires_grad=True)
        )
        # self.linears = nn.ModuleList(nn.Linear(in_dims, out_dims) for i in range(N))
        self.register_parameter("T", nn.Parameter(torch.Tensor([T])))

    def forward(self, x):
        """
        Input: torch.Tensor(B, N, in_dims:25)
        Output: torch.Tensor(B, N)
        B = Batch (2)
        D = Neurons (768)
        M = history/memory_length (25)
        H = Hidden_dims if MLP, or output (4)
        (B, D, M), (M, H, D) -> (B, D, H) x dot weight + bias
        """
        out = self.dropout(x)
        out = self.ln(out)
        # out = [self.linears[i](out[:, i, :]) for i in range(len(self.linears))]
        # print("NLM OUTPUT: ", out.shape, self.w1.shape, self.b1.shape)

        out = torch.einsum("BTDM,MHD->BTDH", out, self.w1) + self.b1
        # out = torch.stack(out, dim=1)
        # print("NLM OUTPUT: ", out.shape)

        out = out.squeeze(-1) / self.T
        # print("NLM OUTPUT: ", out.shape)
        return out


# Continuous Thought Model
class CTM(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.iterations = args.iterations  # Number of CTM iterations (e.g., 20)
        self.vocab_size = args.vocab_size
        self.device = args.device

        # Core dimensions
        self.d_model = args.n_embd  # Model dimension (e.g., 768 for GPT-2)
        self.memory_length = args.memory_length  # Memory sequence length (e.g., 16)
        self.hidden_dims = args.hidden_dimensions  # Hidden dims for NLM (e.g., 4)
        self.dropout = args.dropout
        self.heads = args.n_head  # Number of attention heads (e.g., 12)
        self.out_dims = self.d_model  # Output dims = model dims

        # Synapse Network: Processes concatenated sync + attention features
        # Input: (B*T, d_model + d_model) -> (B*T, d_model)
        # Where sync_action: (B*T, d_model), kv: (B*T, d_model)
        self.synapse_depth = 2
        # SynapseUNet input: (B*T, 2*d_model, 1) -> output: (B*T, d_model, 1)
        self.synnet = SynapseUNet(out_dims=args.n_embd, depth=self.synapse_depth)

        # Neuron-Level Model: Per-neuron MLPs processing memory traces
        # Each of d_model neurons gets its own MLP
        self.neuron_depth = 2
        self.nlm = nn.Sequential(
            nn.Sequential(
                # First NLM: (B*T, memory_length, d_model) -> (B*T, 2*hidden_dims, d_model)
                NLM(
                    in_dims=self.memory_length,  # Input: memory_length (e.g., 16)
                    out_dims=2
                    * self.hidden_dims,  # Output: 2*hidden_dims (e.g., 8) for GLU
                    N=self.d_model,  # Number of parallel MLPs (768)
                    dropout=self.dropout,
                ),
                nn.GLU(),  # GLU halves the dimension: 2*hidden_dims -> hidden_dims
                # Second NLM: (B*T, hidden_dims, d_model) -> (B*T, 2, d_model)
                NLM(
                    in_dims=self.hidden_dims,  # Input: hidden_dims (e.g., 4)
                    out_dims=2,  # Output: 2 for final GLU
                    N=self.d_model,  # Number of parallel MLPs (768)
                    dropout=self.dropout,
                ),
                nn.GLU(),  # Final GLU: 2 -> 1, output: (B*T, 1, d_model)
            )
        )

        # Unused NLM definition (should be removed)
        NLM(in_dims=self.memory_length, out_dims=1, N=self.d_model)

        # GPT backbone (set externally via set_GPT method)
        # self.GPT processes: (B, T) -> (B, T, vocab_size), loss, (B, T, d_model)

        # Final projection to vocabulary
        # Input: (B*T, d_model) -> Output: (B*T, vocab_size)
        self.lm_head = nn.Linear(self.d_model, args.vocab_size, bias=False)

        # Skip connection mixing parameter (learnable scalar)
        # Controls balance between GPT-2 logits and CTM predictions
        self.skip_weight = nn.Parameter(torch.tensor(0.5))  # Shape: scalar

        # Synchronization dimensions for output and action
        self.n_sync_out = int(self.d_model // 2)  # e.g., 384 for output sync
        self.n_sync_act = int(
            self.d_model - self.n_sync_out
        )  # e.g., 384 for action sync

        # Output projection layer (lazy initialization)
        # Input: (B*T, d_model) -> Output: (B*T, vocab_size)
        self.ouput_proj = nn.Sequential(nn.LazyLinear(self.d_model))
        self.init_sync_params("action")
        self.init_sync_params("out")
        self.register_parameter(
            "start_activated_state",
            nn.Parameter(
                torch.zeros((self.d_model)).uniform_(
                    -((1 / (self.d_model)) ** 0.5), (1 / (self.d_model)) ** 0.5
                )
            ),
        )
        self.register_parameter(
            "start_trace",
            nn.Parameter(
                torch.zeros((self.d_model, self.memory_length)).uniform_(
                    -((1 / (self.d_model + self.memory_length)) ** 0.5),
                    (1 / (self.d_model + self.memory_length)) ** 0.5,
                )
            ),
        )
        self.q_proj = nn.Linear(self.n_sync_act, self.d_model)
        self.k_proj = nn.Linear(self.d_model, self.d_model)
        self.v_proj = nn.Linear(self.d_model, self.d_model)
        self.merge_proj = nn.Linear(self.d_model * 2, self.d_model)

        self.GPT = None
        self.ln = nn.LayerNorm(self.vocab_size)
        self.ctmlogitscale = nn.Parameter(torch.tensor(1.0))
        self.currGPTLoss = None

    def set_GPT(self, gpt):
        self.GPT = gpt
        # Freeze the GPT-2 model parameters
        for param in self.GPT.parameters():
            param.requires_grad = False

    def compute_certainty(self, current_prediction):
        """
        Compute the certainty of the current prediction
        Certainty is defined as being 1-normalised entropy.
        input: current_prediction (B,T, D)
        output: certainties (B,T, 2)
        """
        B = current_prediction.size(0)

        # Certainty based on entropy
        pred = F.softmax(current_prediction, dim=-1)
        log_pred = F.log_softmax(current_prediction, dim=-1)
        entropy = -torch.sum(pred * log_pred, dim=-1)
        num_classes = pred.shape[-1]
        max_entropy = torch.log(torch.tensor(num_classes, dtype=torch.float32))

        # Normalize the entropy
        ne = entropy / max_entropy
        # if len(current_prediction.shape) > 2 and reduction == "mean":
        #     ne = ne.flatten(1).mean(-1)

        current_certainty = torch.stack((ne, 1 - ne), -1)
        return current_certainty

    # Compute pairwise synchronization
    def compute_sync(self, activated_state, decay_alpha, decay_beta, r, synch_type):
        if synch_type == "action":  # Get action parameters
            neurons_left = self.action_neuridx_l
            neurons_right = self.action_neuridx_r
        elif synch_type == "out":  # Get input parameters
            neurons_left = self.out_neuridx_l
            neurons_right = self.out_neuridx_r

        left = activated_state[..., neurons_left]
        right = activated_state[..., neurons_right]
        pairwise_product = left * right
        # Compute synchronisation recurrently
        if decay_alpha is None or decay_beta is None:
            decay_alpha = pairwise_product
            decay_beta = torch.ones_like(pairwise_product)
        else:
            decay_alpha = r * decay_alpha + pairwise_product
            decay_beta = r * decay_beta + 1
        sync = decay_alpha * (decay_beta**-0.5)
        return sync, decay_alpha, decay_beta

    # Random Pairing left right divide
    def init_lr_neurons(self, n_sync):
        neuron_indices_left = torch.from_numpy(
            np.random.choice(np.arange(self.d_model), size=n_sync)
        )
        neuron_indices_right = torch.concatenate(
            (torch.from_numpy(np.random.choice(np.arange(self.d_model), size=n_sync)),)
        )
        return neuron_indices_left, neuron_indices_right

    def init_sync_params(self, sync_type: str):
        sync_rep = self.n_sync_act if sync_type == "action" else self.n_sync_out
        l, r = self.init_lr_neurons(sync_rep)

        # register left and right neuron indices
        self.register_buffer(f"{sync_type}_neuridx_l", l)
        self.register_buffer(f"{sync_type}_neuridx_r", r)
        self.register_parameter(
            f"decay_params_{sync_type}",
            nn.Parameter(torch.zeros(sync_rep), requires_grad=True),
        )

    # tracking=False,
    def forward(self, x, targets=None):
        """
        Forward pass of the CTM
        Featurize Logits through GPT2
        Then feed into synapse -> nlm -> synchronisation loop
        x: tokenized words
        y: targets = y
        """
        assert self.GPT is not None
        B, T = x.size(0), x.size(1)
        # if tracking:
        #     # --- Tracking Initialization ---
        #     pre_activations_tracking = []
        #     post_activations_tracking = []
        #     synch_out_tracking = []
        #     synch_action_tracking = []
        #     attention_tracking = []

        """
        Compute the key-value features from the input data using the backbone.
        """
        # featurize gpt
        gptlogits, _, gptfeatures = self.GPT(
            x, targets
        )  # (B,T, Vocab_size), loss, (B, T, D)

        # compute positional encodings and add to properly implement key value pairs for future attention
        pos = torch.arange(0, T, dtype=torch.long, device=x.device)  # shape (T)
        pos_enc = self.GPT.transformer.wpe(pos)
        kv = (gptfeatures + pos_enc).flatten(2)
        k = self.k_proj(kv)
        v = self.v_proj(kv)
        # Initialize predictions and certainties tensors
        predictions = torch.zeros(
            B, T, self.vocab_size, self.iterations, device=x.device
        )  # (B*T, vocab_size, iterations)
        certainties = torch.zeros(
            B, T, 2, self.iterations, device=x.device
        )  # (B*T, 2, iterations)

        # --- Reshape for per-token processing ---
        # Treat the time dimension as part of the batch for the CTM loop
        r_action, r_out = torch.exp(-self.decay_params_action).unsqueeze(0).repeat(
            B, T, 1
        ), torch.exp(-self.decay_params_out).unsqueeze(0).repeat(B, T, 1)

        # Initialize state_trace and activated_state using start parameters
        # Use B,T  for all CTM internal processing
        state_trace = self.start_trace.unsqueeze(0).expand(
            B, T, -1, -1
        )  # (B*T, d_model, memory_length)

        activated_state = self.start_activated_state.unsqueeze(0).expand(
            B, T, -1
        )  # (B*T, d_model)

        # Initialize decay parameters - all should use B,T
        decay_alpha_action = r_action  # (B, T, n_sync_act)
        decay_beta_action = r_action  # (B, T, n_sync_act)
        decay_alpha_out = r_out  # (B,T, n_sync_out)
        decay_beta_out = r_out  # (B, T, n_sync_out)

        # Initialize synchronization
        sync_action, decay_alpha_action, decay_beta_action = self.compute_sync(
            activated_state, None, None, r_action, "action"
        )  # sync_action: (B*T, n_sync_act)

        sync_out, decay_alpha_out, decay_beta_out = self.compute_sync(
            activated_state, None, None, r_out, "out"
        )  # sync_out: (B*T, n_sync_out)

        for stepi in range(self.iterations):
            sync_action, decay_alpha_action, decay_beta_action = self.compute_sync(
                activated_state,
                decay_alpha_action,
                decay_beta_action,
                r_action,
                "action",
            )

            # --- Interact with Data via Attention ---
            q = self.q_proj(sync_action).view(B, T, -1)  # (B*T, 1, d_input)
            attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=False)
            # --- Synaptic Network Processing ---
            synapse_input = torch.cat(
                (attn_out, activated_state), dim=-1
            )  # (B*T, d_input + d_model)
            state = self.synnet(synapse_input)  # (B*T, d_model)

            # --- Update Memory with Pre-activations ---
            state_trace = torch.cat(
                (state_trace[:, :, :, 1:], state.unsqueeze(-1)), dim=-1
            )  # (B*T, d_model, memory_length)

            # --- Neuron-Level Model Processing ---
            # NLM expects (B*T, memory_length, d_model) format
            activated_state = self.nlm(state_trace).squeeze(-1)  # (B*T, d_model)

            # --- Calculate Synchronisation for Output Predictions ---
            sync_out, decay_alpha_out, decay_beta_out = self.compute_sync(
                activated_state,
                decay_alpha_out,
                decay_beta_out,
                r_out,
                synch_type="out",
            )  # sync_out: (B*T, n_sync_out)

            # --- Get CTM Predictions and Certainties ---
            ctm_prediction = self.ouput_proj(sync_out)  # (B,T, d_model)
            ctm_logits = self.GPT.lm_head(
                ctm_prediction
            )  # (B*T, vocab_size) project predictionto vocab size
            current_certainty = self.compute_certainty(ctm_prediction)  # (B*T, 2)

            # --- Skip Connection: Mix GPT-2 logits with CTM predictions ---
            # Skip add gpt2 weights and layernorm
            # Mixing log probs rather than weights\
            alpha = torch.sigmoid(self.skip_weight)
            # print("PCTM, pgpt", p_ctm.shape, p_gpt.shape)
            prediction = (
                alpha * self.ctmlogitscale * ctm_logits + (1 - alpha) * gptlogits
            )

            predictions[..., stepi] = prediction

            certainties[..., stepi] = current_certainty

            # pre_activations_tracking.append(
            #     state_trace[:, :, -1].detach().cpu().numpy()
            # )
            # post_activations_tracking.append(activated_state.detach().cpu().numpy())
            # synch_out_tracking.append(sync_out.detach().cpu().numpy())
            # synch_action_tracking.append(sync_action.detach().cpu().numpy())
            # loss, ce_loss = self.calc_loss(predictions, certainties, targets)
        # print("predictions", predictions.shape)
        return predictions, certainties
        # certainties,
        # (np.array(synch_out_tracking), np.array(synch_action_tracking)),
        # np.array(pre_activations_tracking),
        # np.array(post_activations_tracking),
        # np.array(attention_tracking),

    def calc_loss(self, predictions, certainties, targets):
        """
        Predictions: (B*T, vocab_size, iterations)
        Certainties: (B*T, 2, iterations)
        Targets: (B, T)
        """
        assert targets is not None

        # predictions shape: (B, T, D, iterations)
        # We need to project to vocab size for each iteration
        B, T = predictions.size(0), predictions.size(1)
        # print("losscalc", B, T, predictions.shape, targets.shape)

        # Project predictions to vocabulary size for each iteration
        # Reshape to (B*T * iterations, D) -> apply lm_head -> reshape back

        # # Prepare targets: (B, T) -> (B, T,) and repeat for each iteration
        targets_expanded = targets.unsqueeze(2).expand(
            -1, -1, self.iterations
        )  # (B*T, iterations)
        # Calculate cross entropy loss for each sample and each iteration
        # logits: (B*T, iterations, vocab_size), targets: (B*T, iterations)
        losses = torch.zeros(B, T, self.iterations, device=self.device)
        # print(targets_expanded.shape)
        for i in range(self.iterations):

            pred_flat = predictions[:, :, :, i].view(
                -1, self.vocab_size
            )  # (B*T, vocab_size)
            targets_flat = targets.view(-1)  # (B*T,)
            loss_flat = F.cross_entropy(pred_flat, targets_flat, reduction="none")
            losses[:, :, i] = loss_flat.view(B, T)

        # Find the minimum loss across iterations (dim=-1) for each sample
        min_losses, ce_loss_idx = torch.min(losses, dim=-1)  # (B,T,
        loss = min_losses.mean()  # Average across all samples

        # # For certainty loss, use the iteration with highest certainty
        certainty_scores = certainties[:, 1, :]  # (B*T, iterations) - confidence scores
        _, certainty_loss_idx = torch.max(certainty_scores, dim=-1)  # (B*T,)

        # # Get losses at certainty-based indices
        certainty_losses = losses.gather(1, certainty_loss_idx.unsqueeze(1)).squeeze(
            1
        )  # (B*T,)
        loss_certainty = certainty_losses.mean()

        # # Combined loss
        loss = (loss + loss_certainty) / 2

        return loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Generate new tokens autoregressively from an input sequence of indices.

        Args:
            idx (LongTensor): Input sequence of shape (B, T).
            max_new_tokens (int): Number of new tokens to generate.
            temperature (float): Sampling temperature (default: 1.0).
            top_k (int, optional): If specified, sample from top-k tokens.

        Returns:
            LongTensor: Generated sequence of shape (B, T + max_new_tokens).
        """
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be non-negative")
        if idx.size(1) == 0:
            raise ValueError("Input sequence cannot be empty")

        self.eval()  # Set to evaluation mode
        for _ in range(max_new_tokens):
            # Crop input to block_size
            idx_cond = (
                idx
                if idx.size(1) <= self.args.block_size
                else idx[:, -self.args.block_size :]
            )

            # Forward pass
            logits, _ = self(
                idx_cond, targets=None
            )  # Shape: (B, T, vocab_size, iterations)

            # Select logits from the last time step and last iteration
            logits = logits[:, -1, :, -1] / temperature  # Shape: (B, vocab_size)

            # Optional top-k sampling
            if top_k is not None:
                top_k = min(top_k, logits.size(-1))
                if top_k > 0:
                    v, _ = torch.topk(logits, top_k)
                    logits[logits < v[:, -1:]] = -float("Inf")

            # Convert to probabilities and sample
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)  # Shape: (B, 1)

            # Append new token
            idx = torch.cat((idx, idx_next), dim=1)

        self.train()  # Restore training mode
        return idx


""" EXPERIMENTAL
    def change_skip(self):
        # DYNAMICALLY Update skip weight according to the difference between gptloss and ctmloss
        diff = (
            loss_ce - self.currGPTLoss
        ).detach()  # detach so we don't backprop through the transform
        k = 5.0  # scale factor: sharpness of mapping, tuneable
        eps = 1e-6
        desired_alpha = torch.sigmoid(k * diff)  # scalar or per-batch scalar
        desired_alpha = desired_alpha.clamp(eps, 1 - eps)
        # convert to logit-space to set skip_weight (skip_weight is pre-sigmoid)
        new_skip = torch.log(desired_alpha / (1.0 - desired_alpha))

        # smooth using EMA of previous alpha to avoid abrupt jumps
        ema = 0.9
        current_alpha = torch.sigmoid(self.skip_weight).detach()
        smoothed_alpha = ema * current_alpha + (1 - ema) * desired_alpha
        smoothed_alpha = smoothed_alpha.clamp(eps, 1 - eps)
        new_skip = torch.log(smoothed_alpha / (1.0 - smoothed_alpha))
        # print("NEW skip weight", new_skip)
        # apply directly (in-place)
        with torch.no_grad():
            self.skip_weight.copy_(new_skip)
"""


"""
Experimental DEBUGGING PRINT statements
# # right after computing ctm_logits and gptlogits
# print(
#     "ctm_logits mean/std:",
#     ctm_logits.mean().item(),
#     ctm_logits.std().item(),
# )
# print(
#     "gpt_logits mean/std:", gptlogits.mean().item(), gptlogits.std().item()
# )
# top tokens
# topk_ctm = torch.topk(ctm_logits, 10, dim=-1).indices[0, :10]
# topk_gpt = torch.topk(gptlogits.flatten(0, 1), 10, dim=-1).indices[0, :10]
# print("topk ctm", topk_ctm.tolist(), "topk gpt", topk_gpt.tolist())
# print(
#     "topk ctm",
#     enc.decode(topk_ctm.tolist()),
#     "topk gpt",
#     enc.decode(topk_gpt.tolist()),
# )

# print(
#     "p: ",
#     enc.decode(torch.topk(p, 10, dim=-1).indices[0, :10].tolist()),
# )
"""
