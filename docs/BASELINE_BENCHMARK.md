# Baseline benchmark

Trace-only comparison on committed TriviaQA runs. Higher AUC/catch is better; cost is relative to one greedy local-model answer.

`workspace/combined E4B zero-shot` uses the E4B-trained router weights with each target model's frozen normalization stats. `workspace/combined router` uses the per-model released router weights and is a deployment sanity check, not an out-of-fold estimate. For the 5-fold CV proof layer, use `python analysis/analyze_router.py`.

| model | method | cost | AUC | wrong caught @30% | wrong caught @50% |
|---|---:|---:|---:|---:|---:|
| E4B | first-token logprob | 1.0x | 0.767 | 44% | 69% |
| E4B | mean logprob | 1.0x | 0.657 | 38% | 58% |
| E4B | min logprob | 1.0x | 0.626 | 36% | 57% |
| E4B | workspace E4B zero-shot | 1.0x | 0.795 | 45% | 70% |
| E4B | combined E4B zero-shot | 1.0x | 0.815 | 46% | 71% |
| E4B | workspace router | 1.0x | 0.795 | 45% | 70% |
| E4B | combined router | 1.0x | 0.815 | 46% | 71% |
| 12B | first-token logprob | 1.0x | 0.787 | 50% | 73% |
| 12B | mean logprob | 1.0x | 0.564 | 35% | 57% |
| 12B | min logprob | 1.0x | 0.543 | 33% | 53% |
| 12B | workspace E4B zero-shot | 1.0x | 0.751 | 48% | 73% |
| 12B | combined E4B zero-shot | 1.0x | 0.766 | 52% | 73% |
| 12B | workspace router | 1.0x | 0.835 | 51% | 77% |
| 12B | combined router | 1.0x | 0.855 | 53% | 78% |
| 12B-ablit | first-token logprob | 1.0x | 0.783 | 48% | 74% |
| 12B-ablit | mean logprob | 1.0x | 0.594 | 36% | 57% |
| 12B-ablit | min logprob | 1.0x | 0.562 | 33% | 54% |
| 12B-ablit | workspace E4B zero-shot | 1.0x | 0.767 | 51% | 72% |
| 12B-ablit | combined E4B zero-shot | 1.0x | 0.778 | 51% | 74% |
| 12B-ablit | workspace router | 1.0x | 0.814 | 51% | 74% |
| 12B-ablit | combined router | 1.0x | 0.836 | 54% | 77% |
| 26B-MoE | first-token logprob | 1.0x | 0.768 | 55% | 77% |
| 26B-MoE | mean logprob | 1.0x | 0.604 | 39% | 58% |
| 26B-MoE | min logprob | 1.0x | 0.561 | 34% | 55% |
| 26B-MoE | workspace E4B zero-shot | 1.0x | 0.719 | 50% | 70% |
| 26B-MoE | combined E4B zero-shot | 1.0x | 0.743 | 51% | 73% |
| 26B-MoE | workspace router | 1.0x | 0.760 | 51% | 73% |
| 26B-MoE | combined router | 1.0x | 0.802 | 58% | 77% |
| Qwen-27B | first-token logprob | 1.0x | 0.821 | 60% | 80% |
| Qwen-27B | mean logprob | 1.0x | 0.845 | 59% | 81% |
| Qwen-27B | min logprob | 1.0x | 0.856 | 61% | 84% |
| Qwen-27B | workspace E4B zero-shot | 1.0x | 0.553 | 35% | 56% |
| Qwen-27B | combined E4B zero-shot | 1.0x | 0.664 | 42% | 68% |
| Qwen-27B | workspace router | 1.0x | 0.667 | 43% | 65% |
| Qwen-27B | combined router | 1.0x | 0.860 | 64% | 85% |
