const command = process.argv.includes("info") ? "info" : "system";

const system = {
  available_ram_gb: 8,
  backend: "CUDA",
  cpu_cores: 8,
  cpu_name: "Fixture CPU",
  gpu_count: 1,
  gpu_name: "NVIDIA Fixture",
  gpu_vram_gb: 8,
  gpus: [
    {
      backend: "CUDA",
      name: "NVIDIA Fixture",
      unified_memory: false,
      vram_gb: 8,
    },
  ],
  has_gpu: true,
  total_ram_gb: 16,
  unified_memory: false,
};

if (command === "system") {
  process.stdout.write(JSON.stringify({ system }));
} else {
  process.stdout.write(
    JSON.stringify({
      system,
      models: [
        {
          name: "fixture/distributed-small",
          fit_level: "Perfect",
          run_mode: "GPU",
          runtime: "external GGUF runtime",
          best_quant: "Q8_0",
          estimated_tps: 42,
          measured_tps: 36,
          memory_required_gb: 3,
          usable_context: 8192,
        },
      ],
    }),
  );
}
