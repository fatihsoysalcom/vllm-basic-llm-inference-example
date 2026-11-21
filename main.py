from vllm import LLM, SamplingParams
import time

def main():
    # Model to use. A small model is chosen for quick download and demonstration.
    # For production, you'd typically use larger models like 'meta-llama/Llama-2-7b-hf'.
    model_name = "facebook/opt-125m"

    print(f"Loading model: {model_name}...")
    # Initialize the LLM. This will download the model if not cached.
    # vLLM automatically handles model loading and GPU memory management.
    # `tensor_parallel_size` can be set for multi-GPU setups.
    # `gpu_memory_utilization` can be adjusted for memory management.
    llm = LLM(
        model=model_name,
        tensor_parallel_size=1, # Use 1 GPU for this example
        gpu_memory_utilization=0.9 # Utilize up to 90% of GPU memory
    )

    # Define multiple prompts for generation. vLLM excels at batching
    # these requests for efficient parallel processing on the GPU.
    prompts = [
        "Once upon a time, in a land far, far away,",
        "The quick brown fox jumps over the lazy dog",
        "Explain the concept of large language models in one sentence.",
        "What is the capital of France?",
        "Write a short poem about a cat."
    ]

    # Define sampling parameters for text generation.
    # These parameters control the output's creativity, length, etc.
    sampling_params = SamplingParams(
        temperature=0.7,      # Controls randomness; higher means more random
        top_p=0.95,           # Nucleus sampling; considers top P probability mass
        max_tokens=100,       # Maximum number of tokens to generate
        presence_penalty=0.0, # Penalty for new tokens based on whether they appear in text so far
        frequency_penalty=0.0 # Penalty for new tokens based on their frequency in text so far
    )

    print("\nGenerating responses...")
    start_time = time.time()

    # Perform inference. vLLM's PagedAttention and continuous batching
    # allow for high throughput and low latency, especially with multiple prompts.
    outputs = llm.generate(prompts, sampling_params)

    end_time = time.time()
    print(f"\nGeneration completed in {end_time - start_time:.2f} seconds.")

    # Print the outputs.
    for i, output in enumerate(outputs):
        prompt = output.prompt
        generated_text = output.outputs[0].text # Access the generated text
        print(f"--- Prompt {i+1} ---")
        print(f"Prompt: {prompt!r}")
        print(f"Generated: {generated_text!r}\n")

if __name__ == "__main__":
    main()
