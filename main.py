from vllm import LLM, SamplingParams
import os
import sys

# Set PyTorch memory allocation to avoid fragmentation
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

def main():
    """Main function to run Qwen2.5-3B with vLLM on 8GB VRAM.""" 
    try:
        print("Initializing Qwen2.5-3B-Instruct model...")
        print("This may take a few minutes on first run (downloading model)...")
        
        # Memory-efficient settings for 8GB VRAM (3070Ti) with Qwen2.5-3B
        # This 3B parameter model should fit comfortably in 8GB VRAM (~6GB required)
        llm = LLM(
            model="Qwen/Qwen2.5-3B-Instruct",  # Using Qwen2.5-3B which fits in 8GB VRAM
            max_model_len=2048,  # Reasonable context window for 3B model
            gpu_memory_utilization=0.85,  # Use 85% of GPU memory (safe for 8GB)
            tensor_parallel_size=1,  # Single GPU
            trust_remote_code=True,
            dtype="bfloat16",  # Use bfloat16 for memory efficiency
        )
        
        print("Model loaded successfully!")
        
        # Sampling parameters for text generation
        sampling_params = SamplingParams(
            temperature=0.6,
            top_p=0.95,
            max_tokens=512,  # Can generate up to 512 tokens
        )
        
        # Generate response with the instruct model
        # For Qwen2.5-Instruct, the model handles chat formatting automatically
        prompt = "Hello, how are you?"
        print(f"\nGenerating response for: '{prompt}'")
        print("-" * 50)
        
        outputs = llm.generate([prompt], sampling_params)
        
        # Extract and print the generated text
        for output in outputs:
            generated_text = output.outputs[0].text
            print(f"Response: {generated_text}")
            print("-" * 50)
            
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()