---
title: Car Review Intelligence
emoji: 🚗
colorFrom: red
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: Aspect-level sentiment for car reviews, with the evidence
---

# Car Review Intelligence — v2

Paste an owner review and see the reviewer's opinion on eight aspects (price, performance, comfort, reliability, design, safety, fuel economy, technology), the sentence behind each label, the cars and parts mentioned, and an overall verdict. The CSV tab summarises a whole file of reviews.

- **Model:** [Phoenixak99/car-review-absa](https://huggingface.co/Phoenixak99/car-review-absa), a small encoder fine-tuned on synthetic reviews written and double-checked by a local Qwen3-8B teacher
- **Entities:** GLiNER small (zero-shot)
- **Speed:** well under a second per review on 2 CPU cores

The first load after the Space has been asleep takes about a minute.

Code, data generation and evaluation: [github.com/phoenixak/Analyzing-Car-Reviews-with-LLMs](https://github.com/phoenixak/Analyzing-Car-Reviews-with-LLMs)
