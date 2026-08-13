import os
import sys
import json
import torch

from typing import Union, Literal
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

class ModelHandler:

    def __init__(
            self,
            pretrained_model_name_or_path: Union[str, os.PathLike],
            device: Literal["cuda", "cpu"] = "cuda",
            precision: Literal["bfloat16", "4bit", "8bit", "orig"] = "orig",
            attn: Literal["none", "flash", "sdpa", "eager"] = "none"
            ):
        self.device = device

        # Load the config file.
        config_path = os.path.join(pretrained_model_name_or_path, 'config.json')
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found at {config_path}")
        with open(config_path, 'r') as f:
            config = json.load(f)

        # Determine if the model is Gemma3ForCausalLM.
        # NOTE: The Gemma3 models need attn_implementation="eager" and don't like float16 due to the +/- 2^16 range.
        #       https://old.reddit.com/r/LocalLLaMA/comments/1dsvpp2/thread_on_running_gemma_2_correctly_with_hf/
        architectures = config.get("architectures") or []
        isGemma3 = (
            "Gemma3ForCausalLM" in architectures
            or "gemma3" in (config.get("model_type") or "").lower()
        )

        # 'orig' honors the model's own dtype (per config.json); 'bfloat16'/'4bit'/'8bit' all compute in bfloat16.
        if precision == "orig":
            if "torch_dtype" not in config:
                raise KeyError("The 'torch_dtype' key is missing in the configuration file")
            self.torch_dtype = getattr(torch, config["torch_dtype"])
        else:
            self.torch_dtype = torch.bfloat16

        if isGemma3 and self.torch_dtype == torch.float16:
            print("*** Gemma3ForCausalLM: overriding torch_dtype = bfloat16 (float16 is unsupported) and attn_implementation = 'eager' ***")
            self.torch_dtype = torch.bfloat16
        elif isGemma3:
            print("*** Gemma3ForCausalLM: using attn_implementation = 'eager' ***")

        print(f"Using torch_dtype = {self.torch_dtype}")

        if device not in ("cuda", "cpu"):
            raise RuntimeError(f"The device must be 'cpu' or 'cuda': {device}")

        # Quantization is only supported on 'cuda'.
        if precision in ("4bit", "8bit"):
            if device != "cuda":
                raise RuntimeError(f"Quantization ('{precision}') requires device='cuda', got '{device}'")
            if precision == "4bit":
                print("Using 4-bit quantization")
                self.quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=self.torch_dtype
                )
            else:
                print("Using 8-bit quantization")
                self.quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        else:
            print("Using no quantization")
            self.quantization_config = None


        # Map CLI --attn vocabulary to Transformers' attn_implementation.
        _attn_map = {
            "none": None,
            "flash": "flash_attention_2",
            "sdpa": "sdpa",
            "eager": "eager",
        }
        if attn not in _attn_map:
            raise ValueError(f"Unknown attn value: {attn!r}")
        attn_implementation = _attn_map[attn]

        # Gemma3 requires eager attention; override the user's choice if needed.
        if isGemma3 and attn_implementation != "eager":
            print(f"*** Gemma3ForCausalLM: overriding attn_implementation from {attn_implementation!r} to 'eager' (Gemma3 requires eager) ***")
            attn_implementation = "eager"

        # Guard: if user asked for flash, make sure flash_attn is importable.
        if attn_implementation == "flash_attention_2":
            try:
                import flash_attn  # noqa: F401
            except ImportError:
                raise RuntimeError(
                    "flash-attn is not installed, but --attn flash was requested. "
                    "Install it via: pip install -r requirements-flash.txt, "
                    "or use --attn sdpa / --attn none."
                )

        print(f"Using attn_implementation = {attn_implementation!r}")
        print(f"Loading '{pretrained_model_name_or_path}' model and tokenizer...")
        self.model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path,
            torch_dtype = self.torch_dtype,
            quantization_config = self.quantization_config,
            attn_implementation = attn_implementation,
            device_map = 'auto' if device == "cuda" else 'cpu',
            trust_remote_code=True,
            low_cpu_mem_usage = True,
        )
        self.model.requires_grad_(False)
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path, trust_remote_code=True)

    def get_num_layers(self):
        return len(self.model.model.layers)

    def get_model_type(self):
        return self.model.config.model_type

    def modify_tensor(self, layer_index, direction_matrix):
        assert hasattr(self.model.model, 'layers'), "The model does not have the expected structure."
        direction_matrix = direction_matrix.to(torch.float32)
        if direction_matrix.device != self.model.device:
            direction_matrix = direction_matrix.to(self.model.device)

        # Each vector must have unit norm so V^T.V correctly computes the projection onto the subspace.
        # NOTE: The projection matrix calculation is invariant to the signs of the vectors though...
        direction_matrix = torch.nn.functional.normalize(direction_matrix, p = 2, dim = 1)

        identity_matrix = torch.eye(direction_matrix.size(1), dtype = torch.float32, device = self.model.device)
        projection_matrix = identity_matrix - torch.mm(direction_matrix.t(), direction_matrix)
        weight_matrix = self.model.model.layers[layer_index].mlp.down_proj.weight.data.to(torch.float32)
        weight_matrix = torch.mm(projection_matrix, weight_matrix)
        self.model.model.layers[layer_index].mlp.down_proj.weight = torch.nn.Parameter(weight_matrix.to(self.torch_dtype))

    def modify_tensors(self, direction_matrix, skip_begin_layers, skip_end_layers):
        assert hasattr(self.model.model, 'layers'), "The model does not have the expected structure."
        for layer_index in range(skip_begin_layers, self.get_num_layers() - skip_end_layers):
            self.modify_tensor(layer_index, direction_matrix)

    def save_model_and_tokenizer(self, output_path):
        print(f"Saving modified model + original tokenizer to '{output_path}'... ", end = "")
        sys.stdout.flush()
        self.model.save_pretrained(output_path)
        self.tokenizer.save_pretrained(output_path)
        print("Done.")

    # See: https://github.com/vgel/repeng/blob/main/repeng/extract.py
    def export_gguf(self, directions: list[torch.Tensor | None], path: os.PathLike[str] | str):
        import gguf
        ARCHITECTURE = "controlvector"

        print(f"Initializing GGUFWriter with path: '{path}' and architecture: '{ARCHITECTURE}'")
        writer = gguf.GGUFWriter(path, ARCHITECTURE)

        print(f"- Adding model hint: '{self.get_model_type()}'")
        writer.add_string(f"{ARCHITECTURE}.model_hint", self.get_model_type())

        # Count non-None tensors to determine the layer count
        #non_none_tensors = [tensor for tensor in directions if tensor is not None]
        print(f"- Adding layer count: '{self.get_num_layers()}'")
        writer.add_uint32(f"{ARCHITECTURE}.layer_count", self.get_num_layers())

        # Find the hidden dimension size from the first non-None tensor
        hidden_dimension = next((tensor.shape[1] for tensor in directions if tensor is not None), None)
        if hidden_dimension is None:
            raise ValueError("All tensors are None or no tensor has a second dimension.")

        print(f"Hidden dimension size across tensors: {hidden_dimension}")

        ### @@@ NOTE: Padded with zero tensors to work around llama.cpp code @@@ ###
        for layer, tensor in enumerate(directions):
            """
            if tensor is None:
                # Create a zero tensor with the shape (1, hidden_dimension)
                combined_tensor = torch.zeros((1, hidden_dimension))
                print(f"-- Layer: {layer + 1} is None, using zero tensor of shape: {combined_tensor.shape}")
            else:
                print(f"-- Processing layer: {layer + 1} with tensor of shape: {tensor.shape}")
                if tensor.shape[0] > 1:
                    combined_tensor = torch.sum(tensor, dim=0)
                    print(f"--- Combined vectors for layer {layer + 1} into shape: {combined_tensor.shape}")
                else:
                    combined_tensor = tensor[0]

            writer.add_tensor(f"direction.{layer + 1}", combined_tensor.flatten().numpy())
            """
            if tensor is not None:
                print(f"-- Processing layer: {layer + 1} with tensor of shape: {tensor.shape}")
                if tensor.shape[0] > 1:
                    combined_tensor = torch.sum(tensor, dim=0)
                    print(f"--- Combined vectors for layer {layer + 1} into shape: {combined_tensor.shape}")
                else:
                    combined_tensor = tensor[0]
                writer.add_tensor(f"direction.{layer + 1}", combined_tensor.flatten().numpy())

        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()

        writer.close()

        print("Export completed")

    def export_gguf_conceptors(self, conceptors, means, class_idx, path):
        """
        Exports conceptors and means for a single class in GGUF format.

        Parameters:
            conceptors: List of conceptor matrices for the class [layer_idx] -> ConceptorRepresentation or None
            means: List of mean vectors for the class [layer_idx] -> torch.Tensor(d) or None
            class_idx: The index of the class being exported
            path: Output file path
        """
        import gguf
        from conceptor_analyzer import ConceptorRepresentation, reconstruct_conceptor

        ARCHITECTURE = "conceptor"
        writer = gguf.GGUFWriter(path, ARCHITECTURE)

        print(f"Initializing GGUFWriter with path: '{path}' and architecture: '{ARCHITECTURE}'")

        writer.add_string(f"{ARCHITECTURE}.model_hint", self.get_model_type())
        writer.add_string(f"{ARCHITECTURE}.model_id", os.path.basename(path))

        num_layers = self.get_num_layers()
        writer.add_uint32(f"{ARCHITECTURE}.layer_count", num_layers)

        hidden_dim = None
        for con in conceptors:
            if con is not None:
                if con.is_low_rank:
                    hidden_dim = con.U.shape[0]
                else:
                    hidden_dim = con.full.shape[0]
                break
        if hidden_dim is None:
            raise ValueError("No conceptor found to determine the hidden dimension size")
        writer.add_uint32(f"{ARCHITECTURE}.hidden_dim", hidden_dim)
        print(f"Hidden dimension size: {hidden_dim}")

        print(f"Processing class index: {class_idx}")
        for layer_idx, (con, m) in enumerate(zip(conceptors, means)):
            if con is not None:
                # Always reconstruct to a full matrix so both paths emit
                # an identical GGUF layout.
                if con.is_low_rank:
                    full_matrix = reconstruct_conceptor(con.U, con.s)
                else:
                    full_matrix = con.full
                conceptor_name = f"conceptor.{layer_idx}"
                writer.add_tensor(conceptor_name, full_matrix.cpu().numpy())
                print(f"  - Added conceptor tensor: {conceptor_name} with shape {full_matrix.shape}")
            if m is not None:
                mean_name = f"mean_vector.{layer_idx}"
                writer.add_tensor(mean_name, m.cpu().numpy())
                print(f"  - Added mean vector tensor: {mean_name} with shape {m.shape}")

        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
        writer.close()
        print(f"Exported conceptors and means for class {class_idx} to {path} in GGUF format.")

    def delete(self):
        del self.model
        del self.tokenizer
