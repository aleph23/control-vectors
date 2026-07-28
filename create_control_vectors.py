import argparse
import gc
import sys
import signal
import torch

from model_handler import ModelHandler
from dataset_manager import DatasetManager
from hidden_state_data_manager import HiddenStateDataManager
from direction_analyzer import DirectionAnalyzer
from conceptor_analyzer import ConceptorAnalyzer

def signal_handler(sig, frame):  # @UnusedVariable
    sys.exit(1)

def free_memory():
    gc.collect()
    torch.cuda.empty_cache()

def main(
    model,
    outpath,
    prompts,
    continuations,
    writing_prompts_file,
    num_samples,
    use_system_prompt,
    skip_early_layers,
    skip_late_layers,
    gate,
    conceptors,
    conceptor_aperture,
    center,
    lora,
    lora_method,
    rank,
    variance_thres,
    thres,
    batch_size,
    precision
):
    signal.signal(signal.SIGINT, signal_handler)

    torch.inference_mode()
    torch.set_default_device("cuda")
    torch.set_grad_enabled(False)

    dataset_manager = DatasetManager(
        prompts,
        continuations,
        writing_prompts_file,
        num_samples
    )

    hidden_state_data_manager = HiddenStateDataManager(
        dataset_manager,
        model,
        outpath,
        use_system_prompt,
        batch_size,
        precision
    )

    if use_conceptor:
        print(f"==> Running ConceptorAnalyzer with aperture={conceptor_aperture} center_mode={center_mode}")
        c_analyzer = ConceptorAnalyzer(
            hidden_state_data_manager=hidden_state_data_manager,
            skip_early_layers=skip_early_layers,
            skip_late_layers=skip_late_layers,
            aperture=conceptor_aperture,
            center=center,
            lora=lora,
            lora_method=lora_method,
            rank=rank,
            variance_thres=variance_thres,
            thres=thres
        )

        model_handler = ModelHandler(model, device="cuda")
        num_classes = c_analyzer.num_dataset_types

        for class_idx in range(num_classes):
            conceptor_filename = outpath + f"_conceptor_class_{class_idx}.gguf"
            print(f"Saving computed conceptors and means for class {class_idx} to '{conceptor_filename}'...")
            class_conceptors = c_analyzer.conceptors[class_idx]
            class_means = c_analyzer.means[class_idx]
            model_handler.export_gguf_conceptors(class_conceptors, class_means, class_idx, conceptor_filename)
        model_handler.delete()

    else:
        direction_analyzer = DirectionAnalyzer(
            hidden_state_data_manager,
            skip_early_layers,
            skip_late_layers,
            gate
        )

        for i, direction_matrices_by_class in enumerate(direction_analyzer.direction_matrices):
            if any(direction_matrix_by_layer is not None for direction_matrix_by_layer in direction_matrices_by_class):
                free_memory()
                model_handler = ModelHandler(
                    model, 
                    device="cuda", 
                    precision = bf16
                )
                if i == 0:
                    name = "debias"
                else:
                    name = dataset_manager.class_names[i]

                model_handler.export_gguf(direction_matrices_by_class, outpath + f"_{name}.gguf")

if __name__ == "__main__":
    parser=argparse.ArgumentParser(description="Create and save triad model modifiers aligned on a subjective center and contrasted tendencies")
    parser.add_argument("--model", type=str, required=True, help="Absolute or relative path with filename of our patient")
    parser.add_argument("--outpath", type=str, required=True, help="Where to save the cosmetic prosthetics")
    parser.add_argument("--prompts", type=str, required=True, help="Absolute or relative path to prompt stems file")
    parser.add_argument("--continuations", type=str, required=True, help="Absolute or relative to continuations file")
    parser.add_argument("--writing-prompts-file", type=str, required=True, help="Absolute or relative path to something-Probably find out")
    parser.add_argument("--num-samples", type=int, default=10000, help="The number of prompts to sample per class. Is this not the same thing as batch size?")
    parser.add_argument("--use-system-prompt", action="store_false", default=True, help="Use system prompt. Because this will align with real-world usage.  Find out why not")
    parser.add_argument("--skip-early-layers", type=int, default=0, help="The number of beginning layers to skip.")
    parser.add_argument("--skip-late-layers", type=int, default=1, help="The number of ending layers to skip.")
    parser.add_argument("--contrast-layers", type=[], help="For models with regularly patterned contrast layers, like Gemma4 and Qwen-v35+. First position is the layer number of the first contrasting layer, i.e. '4'. Second position is a string list of the blocks we are probing, i.e. 'attn_k.weight', 'attn_q.weight', 'attn_v.weight' An empty list skips those layers")
    parser.add_argument("--gate", type=float, default=0.3, help = "Noise gate. 0.0 is ungated. 1.0 makes this entire process a pointless hardware stress test, not unlike filling a hole as you dig it, or, more precisely, daydreaming the hole's displacement. Default is a wholy arbitrary 0.3")
    
    parser.add_argument("--conceptors", action="store_true", default=False, help="Use conceptors instead of control vectors")
    parser.add_argument("--conceptor-aperture", type=float, default=0.1, help="Aperture for conceptor if using --use_conceptor. In light capture, aperture is a physical width that affects 1) the amount of light captured and, 2) the 'depth of field' that being the width of the in-focus band on a from-lens z.  What it means here? :shrug:")
    parser.add_argument("--center", type=str, default="none", choices=["none", "local", "baseline"], help="If and how to perform mean-centering during conceptor extraction.  Default skips")
    parser.add_argument("--lora", action="store_true", default=False, help="Use low-rank approximation for conceptors.")
    parser.add_argument("--lora-method", type=str, default=None, choices=["manual", "automatic", "optimal"], help="Method for low-rank approximation ('manual', 'automatic', 'optimal').")
    parser.add_argument("--rank", type=int, default=None, help="Rank for 'manual' low-rank method.")
    parser.add_argument("--variance_thres", type=float, default=None, help="Variance threshold for 'automatic' low-rank method (e.g., 0.99).")
    parser.add_argument("--thres", type=float, default=None, help="Explicit threshold for 'optimal' lora method (e.g., 1e-4).")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for hidden state generation (1 = no batching).")
    parser.add_argument("--precision", default="orig", type=str, choices=["bf16", "4bit", "8bit", "orig"], default="orig", help="Precision for model operations (orig, 4bit, 8bit, bf16 - default: 'orig'inal)")

    args = parser.parse_args()
    main(
        args.model,
        args.outpath,
        args.prompts,
        args.continuations,
        args.writing_prompts_file,
        args.num_samples,
        args.use_system_prompt,
        args.skip_early_layers,
        args.skip_late_layers,
        args.contrast_layers,
        args.gate,
        args.conceptors,
        args.conceptor_aperture,
        args.center,
        args.lora,
        args.lora_method,
        args.rank,
        args.variance_thres,
        args.thres,
        args.batch_size,
        args.precision
    )
