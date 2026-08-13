import os
import sys
import torch

from tqdm import tqdm

from typing import Union, List, Literal

from dataset_manager import DatasetManager
from model_handler import ModelHandler

class HiddenStateDataManager:

    def __init__(
        self,
        dataset_manager: DatasetManager,
        pretrained_model_name_or_path: Union[str, os.PathLike],
        output_path: str,
        use_separate_system_message: bool,
        batch_size: int = 1,
        precision: Literal["bfloat16", "4bit", "8bit", "orig"] = "orig",
        attn: Literal["none", "flash", "sdpa", "eager"] = "none"
    ):
        self.model_handler = None
        self.dataset_hidden_states = []
        self.batch_size = batch_size
        self.precision = precision
        self.attn = attn

        filename = output_path + "_hidden_state_samples.pt"
        cache_loaded = False

        if os.path.exists(filename):
            # NOTE: The cache key is only the output path. Two runs that differ in
            # --num-samples, --no-use-system-prompt, or the model will collide on the
            # same filename. The shape checks below catch class/layer mismatches but
            # not same-shape-different-content collisions. This is a known, accepted
            # limitation — delete the .pt or use a new --outpath to force regeneration.
            data = self._load_and_validate_cache(filename, dataset_manager)
            if data is not None:
                self.dataset_hidden_states = data
                cache_loaded = True
                print(f"Done ({self.get_total_samples()} samples; {self.get_num_layers()} layers).")

        if not cache_loaded:
            self._load_model(pretrained_model_name_or_path)
            dataset_tokens = self._tokenize_datasets(dataset_manager, use_separate_system_message)
            self._generate_hidden_state_samples(dataset_tokens)
            print(f"Saving to '{filename}'... ", end="")
            sys.stdout.flush()
            self.save_hidden_state_samples(filename)
            print("Done.")

    def _load_and_validate_cache(self, file_path: str, dataset_manager: DatasetManager):
        """Load a .pt cache and validate it with shape checks.

        Returns the loaded data on success, None if the cache is stale/malformed
        (triggering regeneration).
        """
        print(f"Loading existing '{file_path}'... ", end="")
        sys.stdout.flush()
        try:
            data = torch.load(file_path, weights_only=True)
        except Exception as e:
            print(f"WARNING: Failed to load cache '{file_path}': {e}. Regenerating...")
            return None

        # Shape checks.
        if not isinstance(data, list) or len(data) == 0:
            print(f"WARNING: Cache '{file_path}' is empty or not a list. Regenerating...")
            return None

        expected_classes = dataset_manager.get_num_classes()
        if len(data) != expected_classes:
            print(f"WARNING: Cache has {len(data)} classes, expected {expected_classes}. Regenerating...")
            return None

        # Every class list is non-empty, and all classes have the same sample count.
        sample_counts = [len(class_samples) for class_samples in data]
        if any(c == 0 for c in sample_counts):
            print(f"WARNING: Cache has empty class(es). Regenerating...")
            return None
        if len(set(sample_counts)) != 1:
            print(f"WARNING: Cache has inconsistent per-class sample counts: {sample_counts}. Regenerating...")
            return None

        # Every sample is a non-empty sequence of tensors, and all samples across all
        # classes have the same layer count.
        first_sample = data[0][0]
        if not isinstance(first_sample, list) or len(first_sample) == 0:
            print(f"WARNING: Cache samples are not non-empty lists. Regenerating...")
            return None
        expected_layers = len(first_sample)

        for class_idx, class_samples in enumerate(data):
            for sample_idx, sample in enumerate(class_samples):
                if not isinstance(sample, list) or len(sample) != expected_layers:
                    print(f"WARNING: Cache sample [{class_idx}][{sample_idx}] has wrong layer count. Regenerating...")
                    return None

        # The trailing dimension (feature size) is consistent across every tensor of a
        # given layer index. Check the first sample of each class.
        for layer_idx in range(expected_layers):
            feature_sizes = set()
            for class_samples in data:
                t = class_samples[0][layer_idx]
                if not isinstance(t, torch.Tensor):
                    print(f"WARNING: Cache element is not a tensor. Regenerating...")
                    return None
                feature_sizes.add(t.shape[-1])
            if len(feature_sizes) != 1:
                print(f"WARNING: Inconsistent feature sizes at layer {layer_idx}: {feature_sizes}. Regenerating...")
                return None

        return data

    def get_datasets(self, layer_index: int) -> List[torch.Tensor]:
        return [torch.stack([sample[layer_index] for sample in dataset]) for dataset in self.dataset_hidden_states]

    def get_differenced_datasets(self, layer_index: int) -> List[torch.Tensor]:
        datasets = self.get_datasets(layer_index)
        return [dataset - datasets[0] for dataset in datasets[1:]]

    def get_num_layers(self) -> int:
        return len(self.dataset_hidden_states[0][0])

    def get_num_dataset_types(self) -> int:
        return len(self.dataset_hidden_states)

    def get_total_samples(self) -> int:
        return sum(len(dataset) for dataset in self.dataset_hidden_states)

    def get_num_features(self, layer_index: int) -> int:
        return self.dataset_hidden_states[0][0][layer_index].shape[-1]

    def load_hidden_state_samples(self, file_path: str) -> None:
        try:
            self.dataset_hidden_states = torch.load(file_path, weights_only=True)
        except Exception as e:
            raise RuntimeError(f"Error loading hidden state samples from {file_path}") from e

    def save_hidden_state_samples(self, file_path: str) -> None:
        try:
            torch.save(self.dataset_hidden_states, file_path)
        except Exception as e:
            raise RuntimeError(f"Error saving hidden state samples to {file_path}") from e

    def _load_model(self, pretrained_model_name_or_path: Union[str, os.PathLike]):
        try:
            self.model_handler = ModelHandler(
                    pretrained_model_name_or_path,
                    device = "cuda",
                    precision = self.precision,
                    attn = self.attn
                )
        except Exception as e:
            raise RuntimeError(f"Failed to load model '{pretrained_model_name_or_path}'") from e

    def _tokenize_datasets(
        self,
        dataset_manager: DatasetManager,
        use_separate_system_message: bool
    ) -> List[List[torch.Tensor]]:
        dataset_tokens = [[] for _ in range(dataset_manager.get_num_classes())]
        try:
            with tqdm(total = dataset_manager.get_total_samples(), desc = "Tokenizing prompts") as bar:
                for i, dataset in enumerate(dataset_manager.datasets):
                    for system_message, prompt in dataset:
                        if use_separate_system_message:
                            conversation = [
                                {"role": "system", "content": system_message},
                                {"role": "user", "content": prompt}
                            ]
                        else:
                            conversation = [{"role": "user", "content": system_message + " " + prompt}]
                        tokens = self.model_handler.tokenizer.apply_chat_template(
                            conversation = conversation,
                            add_generation_prompt = True,
                            return_tensors = "pt"
                        )
                        dataset_tokens[i].append(tokens)
                        bar.update(n = 1)
        except Exception as e:
            raise RuntimeError(f"Error during tokenization") from e
        return dataset_tokens

    def _generate_hidden_state_samples(self, dataset_tokens: List[List[torch.Tensor]]) -> None:
        try:
            num_samples = sum(len(tokens) for tokens in dataset_tokens)
            with tqdm(total=num_samples, desc="Sampling hidden states") as bar:
                for token_list in dataset_tokens:
                    hidden_states = []

                    if self.batch_size <= 1:
                        for tokens in token_list:
                            hidden_states.append(self._generate(tokens))
                            bar.update(n=1)
                    else:
                        # process in batches in original order
                        for i in range(0, len(token_list), self.batch_size):
                            batch_tokens = token_list[i:i + self.batch_size]
                            try:
                                batch_hidden_states = self._generate_batch(batch_tokens)
                                hidden_states.extend(batch_hidden_states)
                                bar.update(n=len(batch_tokens))
                            except Exception as e:
                                raise RuntimeError(f"Error processing batch: {e}") from e

                    self.dataset_hidden_states.append(hidden_states)
        except Exception as e:
            raise RuntimeError(f"Error generating hidden states") from e

    def _generate(self, tokens: torch.Tensor) -> List[torch.Tensor]:
        tokens = tokens.to(self.model_handler.model.device)
        output = self.model_handler.model.generate(
            tokens,
            use_cache=False,
            max_new_tokens=1,
            return_dict_in_generate=True,
            output_hidden_states=True,
            attention_mask=torch.ones(tokens.size(), dtype=torch.long).to(tokens.device),
            pad_token_id=self.model_handler.tokenizer.pad_token_id if self.model_handler.tokenizer.pad_token_id is not None else self.model_handler.tokenizer.eos_token_id
        )
        hidden_states_by_layer = [hidden_state[:, -1, :].squeeze().to('cpu') for hidden_state in
                                  output.hidden_states[-1][:]]
        deltas = [hidden_states_by_layer[i] - hidden_states_by_layer[i - 1] for i in
                  range(1, len(hidden_states_by_layer))]
        return deltas

    def _generate_batch(self, tokens_batch: List[torch.Tensor]) -> List[List[torch.Tensor]]:
        max_length = max(tokens.size(1) for tokens in tokens_batch)
        padded_tokens = []
        attention_masks = []
        pad_token_id = self.model_handler.tokenizer.pad_token_id if self.model_handler.tokenizer.pad_token_id is not None else self.model_handler.tokenizer.eos_token_id
        device = self.model_handler.model.device

        for tokens in tokens_batch:
            seq_len = tokens.size(1)
            padded = torch.full((1, max_length), pad_token_id, dtype=tokens.dtype, device=device)
            # left-padding: pad on the left so the final real token is always at index -1
            padded[:, max_length - seq_len:] = tokens

            # attention mask: 1 for real tokens, 0 for padding
            mask = torch.zeros((1, max_length), dtype=torch.long, device=device)
            mask[:, max_length - seq_len:] = 1

            padded_tokens.append(padded)
            attention_masks.append(mask)

        batch_tokens = torch.cat(padded_tokens, dim=0)
        batch_attention_mask = torch.cat(attention_masks, dim=0)

        output = self.model_handler.model.generate(
            batch_tokens,
            use_cache = False,
            max_new_tokens = 1,
            return_dict_in_generate = True,
            output_hidden_states = True,
            attention_mask = batch_attention_mask,
            pad_token_id = pad_token_id
        )

        batch_deltas = []
        for i in range(len(tokens_batch)):
            hidden_states_by_layer = [hidden_state[i, -1, :].squeeze().to('cpu') for hidden_state in output.hidden_states[-1][:]]
            deltas = [hidden_states_by_layer[j] - hidden_states_by_layer[j - 1] for j in range(1, len(hidden_states_by_layer))]
            batch_deltas.append(deltas)

        return batch_deltas