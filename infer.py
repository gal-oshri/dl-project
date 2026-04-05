"""
Inference entry point for the Compressed Context Memory (CCM) summarization system.

Usage:
    python -m dl_project.infer --weights_path ./compressed_context_model [OPTIONS]

    Options:
        --model_name        Base model (default: meta-llama/Llama-2-7b-chat-hf)
        --weights_path      Path to saved LoRA weights (required)
        --dialogue          Dialogue text to summarize (interactive if omitted)
        --max_new_tokens    Maximum tokens to generate (default: 100)
        --temperature       Sampling temperature (default: 0.7)
        --no_sample         Use greedy decoding instead of sampling
        --dataset_eval      Run evaluation on the test split instead of interactive mode
        --num_examples      Number of examples for dataset_eval (default: 10)
"""

import argparse
import logging
import random
import warnings

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CCM inference / summarization")

    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-2-7b-chat-hf")
    parser.add_argument("--weights_path", type=str, required=True)
    parser.add_argument("--dialogue", type=str, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--no_sample", action="store_true")
    parser.add_argument("--dataset_eval", action="store_true")
    parser.add_argument("--num_examples", type=int, default=10)
    parser.add_argument("--dataset_name", type=str, default="knkarthick/dialogsum")
    parser.add_argument(
        "--lora_targets",
        type=str,
        nargs="+",
        default=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    parser.add_argument(
        "--hf_token",
        type=str,
        default=None,
        help="Hugging Face token for gated models (alternative to huggingface-cli login)",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if args.hf_token:
        from huggingface_hub import login
        print("Logging in to Hugging Face with provided token...")
        login(token=args.hf_token)

    from dl_project.config import CCMConfig, ModelConfig, LoRAConfig
    from dl_project.model.builder import load_trained_model
    from dl_project.inference.generate import generate_summary

    config = CCMConfig(
        model=ModelConfig(model_name=args.model_name),
        lora=LoRAConfig(target_modules=args.lora_targets),
    )

    logger.info("Loading model with trained weights from %s", args.weights_path)
    model, tokenizer, compression_token_id = load_trained_model(config, args.weights_path)
    compression_token = config.model.compression_token

    if args.dataset_eval:
        from datasets import load_dataset

        dataset = load_dataset(args.dataset_name)
        test_set = dataset["test"]
        indices = random.sample(range(len(test_set)), min(args.num_examples, len(test_set)))

        for idx in indices:
            example = test_set[idx]
            print(f"\n{'=' * 80}")
            print(f"Dialogue:\n{example['dialogue']}")
            print(f"\nGround truth:\n{example['summary']}")

            summary = generate_summary(
                model=model,
                tokenizer=tokenizer,
                dialogue=example["dialogue"],
                compression_token=compression_token,
                compression_token_id=compression_token_id,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                do_sample=not args.no_sample,
            )
            print(f"\nGenerated:\n{summary}")

    elif args.dialogue:
        summary = generate_summary(
            model=model,
            tokenizer=tokenizer,
            dialogue=args.dialogue,
            compression_token=compression_token,
            compression_token_id=compression_token_id,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            do_sample=not args.no_sample,
        )
        print(f"\nSummary:\n{summary}")

    else:
        print("Interactive mode — enter a dialogue (Ctrl+D to finish, 'quit' to exit):")
        while True:
            print(f"\n{'=' * 60}")
            print("Enter dialogue (end with an empty line):")
            lines = []
            try:
                while True:
                    line = input()
                    if line.strip() == "":
                        break
                    if line.strip().lower() == "quit":
                        return
                    lines.append(line)
            except EOFError:
                if not lines:
                    return

            dialogue = "\n".join(lines)
            if not dialogue.strip():
                continue

            summary = generate_summary(
                model=model,
                tokenizer=tokenizer,
                dialogue=dialogue,
                compression_token=compression_token,
                compression_token_id=compression_token_id,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                do_sample=not args.no_sample,
            )
            print(f"\nSummary:\n{summary}")


if __name__ == "__main__":
    main()
