import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
from config import GPTConfig
from tqdm import tqdm

# from train import DL

from gpt import GPT


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
        # print("NLM OUTPUT: ", out.shape)

        out = torch.einsum("BDM,MHD->BDH", out, self.w1) + self.b1
        # out = torch.stack(out, dim=1)
        # print("NLM OUTPUT: ", out.shape)

        out = out.squeeze(-1) / self.T
        # print("NLM OUTPUT: ", out.shape)
        return out


# Continuous Thought Model
class CTM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.iterations = 20

        self.device = config.device

        self.d_model = config.n_embd
        self.memory_length = 25
        self.hidden_dims = 4  # Memory hidden dims x
        self.dropout = 0.1
        self.heads = 4
        self.out_dims = self.d_model  # equal to model dimensions

        # Synapse lvl for each neuron
        # Pre-activations R DxM
        self.synapse_depth = 2
        self.synnet = SynapseUNet(out_dims=config.n_embd, depth=self.synapse_depth)
        # Input: post-activation states and Attention output
        # Output: Pre-activations

        # Neuron lvl
        # Each neuron {1 ... D} is given privately parameterized MLP
        self.neuron_depth = 2
        self.nlm = nn.Sequential(
            nn.Sequential(
                NLM(
                    in_dims=self.memory_length,
                    out_dims=2 * self.hidden_dims,
                    N=self.d_model,
                    dropout=self.dropout,
                ),
                nn.GLU(),
                NLM(
                    in_dims=self.hidden_dims,
                    out_dims=2,
                    N=self.d_model,
                    dropout=self.dropout,
                ),
                nn.GLU(),
            )
        )

        NLM(in_dims=self.memory_length, out_dims=1, N=self.d_model)

        # GPT Featurizes text data, we pull from pretrained or untrained weights
        # self.GPT = GPT.from_pretrained("gpt2")

        # Final projection to vocabulary size
        self.lm_head = nn.Linear(self.d_model, config.vocab_size, bias=False)

        # Synchronizations (Post-act)
        self.n_sync_out = int(self.d_model // 2)
        self.n_sync_act = int(self.d_model - self.n_sync_out)

        # Full Run through CTM every 100 tokens or so
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
        self.q_proj = nn.LazyLinear(self.d_model)

    def set_GPT(self, gpt):
        self.GPT = gpt
        # Freeze the GPT-2 model parameters
        for param in self.GPT.parameters():
            param.requires_grad = False

    def compute_certainty(self, current_prediction):
        """
        Compute the certainty of the current prediction
        Certainty is defined as being 1-normalised entropy.
        """
        B = current_prediction.size(0)
        reshaped_pred = current_prediction.reshape([B] + [-1])
        # print("reshaped pred", reshaped_pred.shape)

        # Certainty based on entropy
        pred = F.softmax(reshaped_pred, dim=-1)
        log_pred = F.log_softmax(reshaped_pred, dim=-1)
        entropy = -torch.sum(pred * log_pred, dim=-1)
        num_classes = pred.shape[-1]
        max_entropy = torch.log(torch.tensor(num_classes, dtype=torch.float32))

        # Normalize the entropy
        ne = entropy / max_entropy
        # if len(logits.shape)>2 and reduction == 'mean':
        # ne = normalized_entropy.flatten(1).mean(-1)

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

        left = activated_state[:, neurons_left]
        right = activated_state[:, neurons_right]
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

    def forward(self, x, targets):
        """
        x: tokenized words
        y: targets = y
        """
        B, T = x.size(0), x.size(1)
        # --- Tracking Initialization ---
        pre_activations_tracking = []
        post_activations_tracking = []
        synch_out_tracking = []
        synch_action_tracking = []
        attention_tracking = []

        # Compute Features
        logits, gptloss, kv = self.GPT(x, targets)  # B, T, C
        # print(f"GPT loss: {gptloss.item():.4f}")

        # --- Reshape for per-token processing ---
        # Treat the time dimension as part of the batch for the CTM loop
        kv = kv.view(B * T, 1, self.d_model)  # Shape: (B*T, 1, D)
        batch_size = B * T  # New effective batch size

        r_action, r_out = torch.exp(-self.decay_params_action).unsqueeze(0).repeat(
            batch_size, 1
        ), torch.exp(-self.decay_params_out).unsqueeze(0).repeat(batch_size, 1)
        activated_state = self.start_activated_state.unsqueeze(0).expand(batch_size, -1)
        # print("r_action", r_action.shape, r_out.shape)
        # --- Prepare Storage for Outputs per Iteration ---
        predictions = torch.empty(
            batch_size,
            self.out_dims,
            self.iterations,
            device=self.device,
            dtype=torch.float32,
        )
        certainties = torch.empty(
            batch_size, 2, self.iterations, device=self.device, dtype=torch.float32
        )

        # --- Initialise Recurrent State ---
        state_trace = self.start_trace.unsqueeze(0).expand(
            batch_size, -1, -1
        )  # Shape: (B*T, D, M)
        activated_state = self.start_activated_state.unsqueeze(0).expand(
            batch_size, -1
        )  # Shape: (B*T, D)

        # Init recurrent sync decay values
        decay_alpha_action, decay_beta_action = None, None
        self.decay_params_action.data = torch.clamp(self.decay_params_action, 0, 15)
        self.decay_params_out.data = torch.clamp(self.decay_params_out, 0, 15)
        r_action, r_out = torch.exp(-self.decay_params_action).unsqueeze(0).repeat(
            batch_size, 1
        ), torch.exp(-self.decay_params_out).unsqueeze(0).repeat(batch_size, 1)

        _, decay_alpha_out, decay_beta_out = self.compute_sync(
            activated_state, None, None, r_out, "out"
        )

        for stepi in tqdm(range(self.iterations)):
            sync_action, decay_alpha_action, decay_beta_action = self.compute_sync(
                activated_state,
                decay_alpha_action,
                decay_beta_action,
                r_action,
                "action",
            )
            # print(f"Iteration {stepi}\n")

            q = self.q_proj(sync_action).unsqueeze(1)  # Shape: (B*T, 1, D)
            # print("q: ", q.shape)
            attn_out = F.scaled_dot_product_attention(q, kv, kv, is_causal=False)

            # print("attention", attn_out.squeeze(1).shape)  # (B*T, C)
            logits = torch.cat((attn_out.squeeze(1), activated_state), dim=-1)
            # print("LOGITS SHAPE, (B,T,C)", logits.shape)
            state = self.synnet(logits)  # (B, T, C)

            # state_trace is the history of incoming pre-activations

            state_trace = torch.cat(
                (state_trace[:, :, 1:], state.unsqueeze(-1)), dim=-1
            )
            # print(state_trace.shape)  # Should be batch, d_model, Mem len
            activated_state = self.nlm(state_trace).squeeze(-1)
            # We now have the activated state to feed post-activations
            # --- Calculate Synchronisation for Output Predictions ---
            sync_out, decay_alpha_out, decay_beta_out = self.compute_sync(
                activated_state,
                decay_alpha_out,
                decay_beta_out,
                r_out,
                synch_type="out",
            )

            # --- Get Predictions and Certainties ---
            current_prediction = self.ouput_proj(sync_out)
            # print("curr pred", current_prediction.shape)
            current_certainty = self.compute_certainty(current_prediction)

            predictions[..., stepi] = current_prediction
            certainties[..., stepi] = current_certainty
            pre_activations_tracking.append(
                state_trace[:, :, -1].detach().cpu().numpy()
            )
            post_activations_tracking.append(activated_state.detach().cpu().numpy())
            # attention_tracking.append(attn_weights.detach().cpu().numpy())
            synch_out_tracking.append(sync_out.detach().cpu().numpy())
            synch_action_tracking.append(sync_action.detach().cpu().numpy())

        return (
            predictions,
            certainties,
            (np.array(synch_out_tracking), np.array(synch_action_tracking)),
            np.array(pre_activations_tracking),
            np.array(post_activations_tracking),
            np.array(attention_tracking),
        )

    def calc_loss(self, predictions, certainties, targets):
        """
        Predictions: (B*T, D, iterations)
        Certainties: (B*T, 2, iterations)
        Targets: (B, T)
        """
        assert targets is not None
        # --- Reshape and prepare data ---
        B = targets.size(0)
        T = targets.size(1)

        # predictions shape: (B*T, D, iterations)
        # We need to project to vocab size for each iteration
        batch_size = predictions.size(0)  # B*T

        # Project predictions to vocabulary size for each iteration
        # Reshape to (B*T * iterations, D) -> apply lm_head -> reshape back
        predictions_reshaped = predictions.permute(
            0, 2, 1
        ).contiguous()  # (B*T, iterations, D)
        predictions_reshaped = predictions_reshaped.view(
            -1, self.d_model
        )  # (B*T * iterations, D)

        # Apply language model head to get logits
        logits = self.lm_head(predictions_reshaped)  # (B*T * iterations, vocab_size)
        logits = logits.view(
            batch_size, self.iterations, -1
        )  # (B*T, iterations, vocab_size)

        # Prepare targets: (B, T) -> (B*T,) and repeat for each iteration
        targets_flat = targets.view(-1)  # (B*T,)
        targets_expanded = targets_flat.unsqueeze(1).expand(
            -1, self.iterations
        )  # (B*T, iterations)

        # Calculate cross entropy loss for each sample and each iteration
        # logits: (B*T, iterations, vocab_size), targets: (B*T, iterations)
        losses = torch.zeros(batch_size, self.iterations, device=logits.device)

        for i in range(self.iterations):
            losses[:, i] = F.cross_entropy(
                logits[:, i, :], targets_expanded[:, i], reduction="none"
            )

        # Find the minimum loss across iterations (dim=-1) for each sample
        min_losses, ce_loss_idx = torch.min(losses, dim=-1)  # (B*T,), (B*T,)
        loss_ce = min_losses.mean()  # Average across all samples

        # For certainty loss, use the iteration with highest certainty
        certainty_scores = certainties[:, 1, :]  # (B*T, iterations) - confidence scores
        _, certainty_loss_idx = torch.max(certainty_scores, dim=-1)  # (B*T,)

        # Get losses at certainty-based indices
        certainty_losses = losses.gather(1, certainty_loss_idx.unsqueeze(1)).squeeze(
            1
        )  # (B*T,)
        loss_certainty = certainty_losses.mean()

        # Combined loss
        loss = (loss_ce + loss_certainty) / 2

        # print(f"Cross entropy loss (min across iterations): {loss_ce.item():.4f}")
        # print(f"Certainty-based loss: {loss_certainty.item():.4f}")
        # print(f"Combined loss: {loss.item():.4f}")

        return loss, ce_loss_idx

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        self.eval()  # Set the model to evaluation mode
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = (
                idx
                if idx.size(1) <= self.config.block_size
                else idx[:, -self.config.block_size :]
            )
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond, targets=None)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        self.train()  # Set the model back to training mode
        return idx

    def plot_neural_dynamics(
        post_activations_history,
        N_to_plot,
        save_location,
        axis_snap=False,
        N_per_row=5,
        which_neurons_mid=None,
        mid_colours=None,
        use_most_active_neurons=False,
    ):
        assert (
            N_to_plot % N_per_row == 0
        ), f"For nice visualisation, N_to_plot={N_to_plot} must be a multiple of N_per_row={N_per_row}"
        assert post_activations_history.shape[-1] >= N_to_plot
        figscale = 2
        aspect_ratio = 3
        mosaic = (
            np.array([[f"{i}"] for i in range(N_to_plot)])
            .flatten()
            .reshape(-1, N_per_row)
        )
        fig_synch, axes_synch = plt.subplot_mosaic(
            mosaic=mosaic,
            figsize=(
                figscale * mosaic.shape[1] * aspect_ratio * 0.2,
                figscale * mosaic.shape[0] * 0.2,
            ),
        )
        fig_mid, axes_mid = plt.subplot_mosaic(
            mosaic=mosaic,
            figsize=(
                figscale * mosaic.shape[1] * aspect_ratio * 0.2,
                figscale * mosaic.shape[0] * 0.2,
            ),
            dpi=200,
        )

        palette = sns.color_palette("husl", 8)

        which_neurons_synch = np.arange(N_to_plot)
        # which_neurons_mid = np.arange(N_to_plot, N_to_plot*2) if post_activations_history.shape[-1] >= 2*N_to_plot else np.random.choice(np.arange(post_activations_history.shape[-1]), size=N_to_plot, replace=True)
        random_indices = np.random.choice(
            np.arange(post_activations_history.shape[-1]),
            size=N_to_plot,
            replace=post_activations_history.shape[-1] < N_to_plot,
        )
        if use_most_active_neurons:
            metric = (
                np.abs(np.fft.rfft(post_activations_history, axis=0))[3:].mean(0).std(0)
            )
            random_indices = np.argsort(metric)[-N_to_plot:]
            np.random.shuffle(random_indices)
        which_neurons_mid = (
            which_neurons_mid if which_neurons_mid is not None else random_indices
        )

        if mid_colours is None:
            mid_colours = [palette[np.random.randint(0, 8)] for ndx in range(N_to_plot)]
        with tqdm(
            total=N_to_plot, initial=0, leave=False, position=1, dynamic_ncols=True
        ) as pbar_inner:
            pbar_inner.set_description("Plotting neural dynamics")
            for ndx in range(N_to_plot):

                ax_s = axes_synch[f"{ndx}"]
                ax_m = axes_mid[f"{ndx}"]

                traces_s = post_activations_history[:, :, which_neurons_synch[ndx]].T
                traces_m = post_activations_history[:, :, which_neurons_mid[ndx]].T
                c_s = palette[np.random.randint(0, 8)]
                c_m = mid_colours[ndx]

                for traces_s_here, traces_m_here in zip(traces_s, traces_m):
                    ax_s.plot(
                        np.arange(len(traces_s_here)),
                        traces_s_here,
                        linestyle="-",
                        color=c_s,
                        alpha=0.05,
                        linewidth=0.6,
                    )
                    ax_m.plot(
                        np.arange(len(traces_m_here)),
                        traces_m_here,
                        linestyle="-",
                        color=c_m,
                        alpha=0.05,
                        linewidth=0.6,
                    )

                ax_s.plot(
                    np.arange(len(traces_s[0])),
                    traces_s[0],
                    linestyle="-",
                    color="white",
                    alpha=1,
                    linewidth=2.5,
                )
                ax_s.plot(
                    np.arange(len(traces_s[0])),
                    traces_s[0],
                    linestyle="-",
                    color=c_s,
                    alpha=1,
                    linewidth=1.3,
                )
                ax_s.plot(
                    np.arange(len(traces_s[0])),
                    traces_s[0],
                    linestyle="-",
                    color="black",
                    alpha=1,
                    linewidth=0.3,
                )
                ax_m.plot(
                    np.arange(len(traces_m[0])),
                    traces_m[0],
                    linestyle="-",
                    color="white",
                    alpha=1,
                    linewidth=2.5,
                )
                ax_m.plot(
                    np.arange(len(traces_m[0])),
                    traces_m[0],
                    linestyle="-",
                    color=c_m,
                    alpha=1,
                    linewidth=1.3,
                )
                ax_m.plot(
                    np.arange(len(traces_m[0])),
                    traces_m[0],
                    linestyle="-",
                    color="black",
                    alpha=1,
                    linewidth=0.3,
                )
                if axis_snap and np.all(np.isfinite(traces_s[0])):
                    ax_s.set_ylim(
                        [
                            np.min(traces_s[0]) - np.ptp(traces_s[0]) * 0.05,
                            np.max(traces_s[0]) + np.ptp(traces_s[0]) * 0.05,
                        ]
                    )
                    ax_m.set_ylim(
                        [
                            np.min(traces_m[0]) - np.ptp(traces_m[0]) * 0.05,
                            np.max(traces_m[0]) + np.ptp(traces_m[0]) * 0.05,
                        ]
                    )

                ax_s.grid(False)
                ax_m.grid(False)
                ax_s.set_xlim([0, len(traces_s[0]) - 1])
                ax_m.set_xlim([0, len(traces_m[0]) - 1])

                ax_s.set_xticklabels([])
                ax_s.set_yticklabels([])

                ax_m.set_xticklabels([])
                ax_m.set_yticklabels([])
                pbar_inner.update(1)
        fig_synch.tight_layout(pad=0.05)
        fig_mid.tight_layout(pad=0.05)
        if save_location is not None:
            fig_synch.savefig(f"{save_location}/neural_dynamics_synch.pdf", dpi=200)
            fig_synch.savefig(f"{save_location}/neural_dynamics_synch.png", dpi=200)
            fig_mid.savefig(f"{save_location}/neural_dynamics_other.pdf", dpi=200)
            fig_mid.savefig(f"{save_location}/neural_dynamics_other.png", dpi=200)
            plt.close(fig_synch)
            plt.close(fig_mid)
        return fig_synch, fig_mid, which_neurons_mid, mid_colours
