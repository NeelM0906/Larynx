"""Larynx training worker — LoRA fine-tuning orchestration.

See ORCHESTRATION-M7.md for the design. Public surface is intentionally
small:

- :class:`TrainingWorkerServer` handles the
  :class:`TrainLoraRequest` streaming RPC via the shared IPC.
- :class:`TrainingJobRunner` owns a single job's state machine.
"""
