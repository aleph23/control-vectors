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


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Create and save triad model modifiers aligned on a subjective center and contrasted tendencies")
    parser.add_argument("--model", type=str, required=True, help="Absolute or relative path with filename of our patient")
    parser.add_argument("--outpath", type=str, required=True, help="Where to save the cosmetic prosthetics")
    parser.add_argument("--prompts", type=str, required=True, help="Absolute or relative path to prompt stems file")
    parser.add_argument("--continuations", type=str, required=True, help="Absolute or relative path to the continuations JSON file")
    parser.add_argument("--writing-prompts-file", type=str, required=True, help="Absolute or relative path to the newline-delimited creative-writing prompts file (data/writing_prompts.txt)")
    parser.add_argument("--num-samples", type=int, default=10000, help="The total number of prompt samples to generate, split evenly across classes (baseline + the two continuation classes).")
    parser.add_argument("--no-use-system-prompt", dest="use_system_prompt", action="store_false", default=True, help="Disable the separate system prompt. The system prompt is used by default, because it aligns with real-world usage.")
    parser.add_argument("--skip-early-layers", type=int, default=1, help="The number of beginning layers to skip.")
    parser.add_argument("--skip-late-layers", type=int, default=1, help="The number of ending layers to skip.")
    # parser.add_argument("--contrast-layers", nargs='*', default=[], type=int, help="For models with regularly patterned contrast layers, like Gemma4 and Qwen-v35+. First position is the layer number of the first contrasting layer, i.e. '4'. Second position is a string list of the blocks we are probing, i.e. 'attn_k.weight', 'attn_q.weight', 'attn_v.weight' An empty list skips those layers")
    parser.add_argument("--gate", type=float, default=0.3, help="Noise gate. 0.0 is ungated. 1.0 makes this entire process a pointless hardware stress test, not unlike daydreaming you dug a hole and then filled it. Default is a wholy arbitrary 0.3")

    parser.add_argument("--conceptors", action="store_true", default=False, help="Use conceptors instead of control vectors")
    parser.add_argument("--conceptor-aperture", type=float, default=0.1, help="Aperture for conceptor if using --use_conceptor. In light capture, aperture is a physical width that affects 1) the amount of light captured and, 2) the 'depth of field' that being the width of the in-focus band on a from-lens z.  What it means here? :shrug:")
    parser.add_argument("--center", type=str, default="none", choices=["none", "local", "baseline"], help="If and how to perform mean-centering during conceptor extraction.  Default skips")
    parser.add_argument("--lora", action="store_true", default=False, help="Use low-rank approximation for conceptors.")
    parser.add_argument("--lora-method", type=str, default=None, choices=["manual", "automatic", "optimal"], help="Method for low-rank approximation ('manual', 'automatic', 'optimal').")
    parser.add_argument("--rank", type=int, default=None, help="Rank for 'manual' low-rank method.")
    parser.add_argument("--variance-thres", type=float, default=None, help="Variance threshold for 'automatic' low-rank method (e.g., 0.99).")
    parser.add_argument("--thres", type=float, default=None, help="Explicit threshold for 'optimal' lora method (e.g., 1e-4).")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for hidden state generation (1 = no batching).")
    parser.add_argument("--precision", default="orig", type=str, choices=["bfloat16", "4bit", "8bit", "orig"], help="Precision for model operations (orig, 4bit, 8bit, bfloat16 - default: 'orig'inal)")
    parser.add_argument("--attn", type=str, default="none", choices=["none", "flash", "sdpa", "eager"], help="Attention implementation. 'none' lets Transformers auto-select.")
    return parser.parse_args(argv)


def signal_handler(sig, frame):  # @UnusedVariable
    sys.exit(1)


def free_memory():
    gc.collect()
    torch.cuda.empty_cache()


def main(args):
    signal.signal(signal.SIGINT, signal_handler)

    torch.set_grad_enabled(False)
    torch.set_default_device("cuda")

    with torch.inference_mode():
        dataset_manager = DatasetManager(
            args.prompts,
            args.continuations,
            args.writing_prompts_file,
            args.num_samples
        )

        hidden_state_data_manager = HiddenStateDataManager(
            dataset_manager,
            args.model,
            args.outpath,
            args.use_system_prompt,
            args.batch_size,
            args.precision,
            args.attn
        )

        if args.conceptors:
            print(f"==> Running ConceptorAnalyzer with aperture={args.conceptor_aperture} center={args.center}")
            c_analyzer = ConceptorAnalyzer(
                hidden_state_data_manager=hidden_state_data_manager,
                skip_early_layers=args.skip_early_layers,
                skip_late_layers=args.skip_late_layers,
                aperture=args.conceptor_aperture,
                center=args.center,
                lora=args.lora,
                lora_method=args.lora_method,
                rank=args.rank,
                variance_thres=args.variance_thres,
                thres=args.thres
            )

            free_memory()
            model_handler = ModelHandler(args.model, device="cuda", precision=args.precision, attn=args.attn)
            num_classes = c_analyzer.num_dataset_types

            for class_idx in range(num_classes):
                name = "debias" if class_idx == 0 else dataset_manager.class_names[class_idx]
                conceptor_filename = args.outpath + f"_conceptor_{name}.gguf"
                print(f"Saving computed conceptors and means for class {class_idx} to '{conceptor_filename}'...")
                class_conceptors = c_analyzer.conceptors[class_idx]
                class_means = c_analyzer.means[class_idx]
                model_handler.export_gguf_conceptors(class_conceptors, class_means, class_idx, conceptor_filename)
            model_handler.delete()

        else:
            direction_analyzer = DirectionAnalyzer(
                hidden_state_data_manager,
                args.skip_early_layers,
                args.skip_late_layers,
                args.gate
            )

            for i, direction_matrices_by_class in enumerate(direction_analyzer.direction_matrices):
                if any(direction_matrix_by_layer is not None for direction_matrix_by_layer in direction_matrices_by_class):
                    free_memory()
                    model_handler = ModelHandler(
                        args.model,
                        device="cuda",
                        precision=args.precision,
                        attn=args.attn
                    )
                    if i == 0:
                        name = "debias"
                    else:
                        name = dataset_manager.class_names[i]

                    model_handler.export_gguf(direction_matrices_by_class, args.outpath + f"_{name}.gguf")
                    model_handler.delete()


if __name__ == "__main__":
    main(parse_args())