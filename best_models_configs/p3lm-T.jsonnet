local seed = std.parseJson(std.extVar('seed'));
local cuda_device = std.parseJson(std.extVar('CUDA_DEVICE'));
{
  "data_loader": {
    "batch_sampler": {
      "batch_size": 1,
      "type": "bucket"
    },
    "max_instances_in_memory": 1000,
    "num_workers": 5,
    "start_method": "fork"
  },
  "dataset_reader": {
    "create_prefixes": null,
    "create_task_field": true,
    "empty_action_text": "<|noact|>",
    "end_action_text": "<|eact|>",
    "end_of_action_text": "<|eoa|>",
    "file_reader": {
      "audio_features": false,
      "i3d_features": false,
      "observation_length": 4,
      "observation_offset": -2,
      "prefix_task_descriptions_to_steps": false,
      "read_features": true,
      "read_frames": false,
      "resnet_features": false,
      "s3d_features": true,
      "type": "crosstask"
    },
    "move_last_prefix_action_to_begining": false,
    "observation_type": "pre",
    "start_action_text": "<|sact|>",
    "type": "closed-domain-procedural-planning"
  },
  "evaluate_on_test": true,
  "model": {
    "max_steps": 4,
    "min_steps": 4,
    "model_name": "gpt2",
    "new_tokens": {
      "empty_action_text": "<|noact|>",
      "end_action_text": "<|eact|>",
      "end_of_action_text": "<|eoa|>",
      "start_action_text": "<|sact|>"
    },
    "new_tokens_loss_weight": 1,
    "observation_loss": "mse",
    "observation_loss_weight": 1,
    "observation_type": "pre",
    "observations_encoder": {
      "contextualizer": {
        "attention_dropout": 0.2,
        "hidden_dropout": 0.1,
        "input_size": 1024,
        "num_attention_heads": 8,
        "num_hidden_layers": 8,
        "output_size": 1024,
        "type": "transformer"
      },
      "projector": {
        "feedforward": {
          "activations": [
            "relu",
            "linear"
          ],
          "dropout": 0.1,
          "hidden_dims": [
            1536,
            768
          ],
          "input_dim": 1024,
          "num_layers": 2
        },
        "type": "feedforward"
      },
      "type": "contextualize-and-project"
    },
    "random_lm_weights": false,
    "random_obs_masking_ratio": 0,
    "task_embedding": false,
    "top_k": 10,
    "type": "hfp3lm"
  },
  "numpy_seed": seed,
  "pytorch_seed": seed,
  "random_seed": seed,
  "test_data_path": ".data/crosstask/test.txt",
  "train_data_path": ".data/crosstask/@(train|validation).txt",
  "trainer": {
    "callbacks": [
      "track_epoch_callback",
      {
        "type": "should_validate_callback",
        "validation_interval": 1,
        "validation_start": 10
      },
      {
        "distribution_interval": null,
        "save_model_archive": true,
        "should_log_learning_rate": false,
        "should_log_parameter_statistics": false,
        "sub_callbacks": [
          {
            "priority": 100,
            "type": "log_best_validation_metrics"
          }
        ],
        "summary_interval": 100,
        "type": "wandb_allennlp",
        "watch_model": false
      }
    ],
    "checkpointer": {
      "keep_most_recent_by_count": 1,
      "type": "default"
    },
    "cuda_device": cuda_device,
    "grad_norm": 1,
    "num_epochs": 9,
    "num_gradient_accumulation_steps": 4,
    "optimizer": {
      "lr": 5e-05,
      "parameter_groups": [
        [
          [
            "vision_model\\..*"
          ],
          {
            "requires_grad": false
          }
        ]
      ],
      "type": "huggingface_adamw",
      "weight_decay": 0
    },
    "patience": 4,
    "validation_metric": "+next_action_accuracy"
  },
  "type": "train_test_log_to_wandb",
  "validation_data_path": ".data/crosstask/test.txt",
  "validation_dataset_reader": {
    "create_prefixes": "all",
    "create_task_field": true,
    "empty_action_text": "<|noact|>",
    "end_action_text": "<|eact|>",
    "end_of_action_text": "<|eoa|>",
    "file_reader": {
      "audio_features": false,
      "i3d_features": false,
      "observation_length": 4,
      "observation_offset": -2,
      "prefix_task_descriptions_to_steps": false,
      "read_features": true,
      "read_frames": false,
      "resnet_features": false,
      "s3d_features": true,
      "type": "crosstask"
    },
    "minimum_prefix_length": 1,
    "move_last_prefix_action_to_begining": false,
    "observation_type": "pre",
    "start_action_text": "<|sact|>",
    "type": "closed-domain-procedural-planning"
  },
  "vocabulary": {
    "directory": ".data/crosstask/vocabs/closed_domain_gpt2.tar.gz",
    "type": "from_files"
  }
}