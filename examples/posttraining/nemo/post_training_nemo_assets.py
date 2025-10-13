# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


def run(cmd, shell=False, check=True, cwd=os.getcwd()):
    """Utility function to run commands with consistent logging."""
    print(f"Running: {cmd if isinstance(cmd, str) else ' '.join(cmd)}")
    result = subprocess.run(cmd, shell=shell, check=check, cwd=cwd)
    return result


def copy_files_to_directory(source_path, dest_directory, copy_contents=True):
    """
    Generic function to copy files or directories to a destination directory.

    Args:
        source_path (str or Path): Path to source file or directory
        dest_directory (str or Path): Destination directory path
        copy_contents (bool): If True and source is a directory, copy its contents.
                             If False, copy the directory itself.
    """
    source_path = Path(source_path)
    dest_dir = Path(dest_directory)

    if not source_path.exists():
        print(f"Warning: Source path {source_path} does not exist. Skipping copy operation.")
        return

    # Create destination directory if it doesn't exist
    dest_dir.mkdir(parents=True, exist_ok=True)

    if source_path.is_file():
        # Copy single file
        dest_file = dest_dir / source_path.name
        shutil.copy2(source_path, dest_file)
        print(f"Copied file: {source_path.name} to {dest_dir}")
    elif source_path.is_dir():
        if copy_contents:
            # Copy all contents from source directory to destination
            for item in source_path.iterdir():
                if item.is_file():
                    dest_file = dest_dir / item.name
                    shutil.copy2(item, dest_file)
                    print(f"Copied file: {item.name}")
                elif item.is_dir():
                    dest_subdir = dest_dir / item.name
                    shutil.copytree(item, dest_subdir, dirs_exist_ok=True)
                    print(f"Copied directory: {item.name}")
            print(f"Successfully copied contents from {source_path} to {dest_dir}")
        else:
            # Copy the entire directory as a subdirectory
            dest_subdir = dest_dir / source_path.name
            shutil.copytree(source_path, dest_subdir, dirs_exist_ok=True)
            print(f"Copied directory: {source_path} to {dest_subdir}")
    else:
        print(f"Warning: {source_path} is neither a file nor a directory.")


class PostTrainNemoAssets:
    """
    A class to manage the complete post-training pipeline for Nemo assets.

    This class provides a structured way to handle the entire workflow from
    downloading assets to running inference, with configurable parameters
    and state tracking.
    """

    def __init__(
        self,
        max_iters=None,
        checkpoint_save_iter=None,
        datasets_dir="datasets/cosmos_nemo_assets",
        checkpoints_base_dir="/root/.cache/huggingface/hub",
        output_dir="outputs/video2world_cosmos_nemo_posttraining",
        temp_checkpoint_base_dir="/tmp/imaginaire4-output/cosmos_diffusion_v2/cosmos_nemo_posttraining/predict2_video2world_2b_cosmos_nemo_assets/checkpoints",
        prompt="A video of sks teal robot.",
        experiment_name="predict2_video2world_training_2b_cosmos_nemo_assets",
    ):
        """
        Initialize the PostTrainNemoAssets pipeline.

        Args:
            max_iters (int): Maximum training iterations (defaults to env MAX_ITERS or 200)
            checkpoint_save_iter (int): Frequency to save checkpoint
            datasets_dir (str): Directory for storing datasets
            checkpoints_base_dir (str): Base directory for model checkpoints
            output_dir (str): Directory for inference outputs
            temp_checkpoint_base_dir (str): Base temporary directory for training checkpoints
            prompt (str): Text prompt for T5 embeddings
            experiment_name (str): Name of the experiment configuration
        """
        # Environment variable MAX_ITERS takes precedence over command line argument
        self.max_iters = int(os.getenv("MAX_ITERS", max_iters or 1))
        self.checkpoint_save_iter = checkpoint_save_iter
        self.datasets_dir = datasets_dir
        self.checkpoints_base_dir = checkpoints_base_dir
        self.output_dir = output_dir
        self.temp_checkpoint_base_dir = temp_checkpoint_base_dir
        self.prompt = prompt
        self.experiment_name = experiment_name

        # Derived paths
        self.videos_dir = f"{self.datasets_dir}/videos"
        self.nemo_data_dir = f"{self.datasets_dir}/nemo_diffusion_example_data"
        self.trained_checkpoints_dir = f"{self.checkpoints_base_dir}/trained"
        self.base_checkpoints_dir = f"{self.checkpoints_base_dir}/models--nvidia--Cosmos-Predict2.5-2B/snapshots/a64a214a5ff6937c35ac32a41a7922442ccdf774/d20b7120-df3e-4911-919d-db6e08bad31c"

        # Build temp checkpoint dir with proper 9-digit formatting
        self.temp_checkpoint_dir = f"{self.temp_checkpoint_base_dir}/iter_{self.max_iters:09d}"

        # Pipeline state tracking
        self.pipeline_state = {
            "assets_downloaded": False,
            "assets_copied": False,
            "training_completed": False,
            "checkpoints_converted": False,
            "inference_completed": False,
        }

        print(f"PostTrainNemoAssets initialized with max_iters={self.max_iters}")
        print(f"Datasets dir: {self.datasets_dir}")
        print(f"Checkpoints dir: {self.checkpoints_base_dir}")
        print(f"Temp checkpoint dir: {self.temp_checkpoint_dir}")

    def download_assets(self):
        """Download NeMo assets from HuggingFace."""
        if os.path.exists(self.datasets_dir):
            return self
        print("Downloading assets...")
        os.makedirs(self.datasets_dir, exist_ok=True)
        run(
            [
                "huggingface-cli",
                "download",
                "nvidia/Cosmos-NeMo-Assets",
                "--repo-type",
                "dataset",
                "--local-dir",
                self.datasets_dir,
                "--include",
                "*.mp4*",
            ]
        )
        # Move and organize files
        if os.path.exists(self.nemo_data_dir):
            self.pipeline_state["assets_copied"] = True
            shutil.move(self.nemo_data_dir, self.videos_dir)
        print("Assets downloaded and organized.")
        self.pipeline_state["assets_downloaded"] = True
        return self

    def list_assets(self):
        """List contents of the assets directory."""
        print("Listing assets directory:")
        run(["ls", "-l", self.datasets_dir])
        return self

    def copy_nemo_assets_to_videos(self):
        """Copy contents from nemo_diffusion_example_data to videos directory."""
        copy_files_to_directory(self.nemo_data_dir, self.videos_dir, copy_contents=True)
        self.pipeline_state["assets_copied"] = True
        return self

    def create_prompts_for_nemo_assets(self):
        """Create prompts for the nemo assets."""
        print("Creating prompts for the nemo assets...")
        run(
            [
                sys.executable,
                "scripts/create_prompts_for_nemo_assets.py",
                "--dataset_path",
                self.datasets_dir,
                "--prompt",
                self.prompt,
            ]
        )
        return self

    def setup_environment_variables(self):
        """Set up required environment variables for training."""
        env_vars = {
            "COSMOS_INTERNAL": "0",
            "COSMOS_REASON1_DIR": "checkpoints/nvidia/Cosmos-Reason1-7B",
            "COSMOS_QWEN_2PT5_VL_7B_INSTRUCT_PATH": "checkpoints/nvidia/Cosmos-Reason1-7B",
            "COSMOS_WAN2PT1_VAE_PATH": f"{self.base_checkpoints_dir}/tokenizer.pth",
            "COSMOS_WAN2PT1_VAE_MEAN_STD_PATH": f"{self.base_checkpoints_dir}/mean_std.pt",
        }

        for key, value in env_vars.items():
            os.environ[key] = value
            print(f"Set {key}={value}")

        return self

    def run_training(self):
        """Execute the training process."""
        print("Starting training...")

        # Set up environment
        self.setup_environment_variables()

        # Build training command
        base_cmd = "torchrun --nproc_per_node=8 --master_port=12341"
        script = "-m scripts.train"
        config = "cosmos_predict2/_src/predict2/configs/video2world/config.py"

        # Build command parts for better readability
        cmd_parts = [
            f"{base_cmd} {script} --config={config} --",
            f"experiment={self.experiment_name}",
            f"trainer.max_iter={self.max_iters}",
            f"checkpoint.save_iter={self.checkpoint_save_iter}",
            "job.wandb_mode=disabled",
        ]
        command = " ".join(cmd_parts)
        run(command, shell=True)
        print("Training completed.")
        self.pipeline_state["training_completed"] = True
        return self

    def list_checkpoints(self):
        """List the trained checkpoints."""
        print("Listing trained checkpoints:")
        # List the base checkpoints directory
        run(["ls", "-l", self.temp_checkpoint_base_dir])

        # List specific iteration checkpoint directory
        if os.path.exists(self.temp_checkpoint_dir):
            run(["ls", "-l", self.temp_checkpoint_dir])
        else:
            print(f"Warning: Checkpoint directory {self.temp_checkpoint_dir} not found")
        return self

    def convert_distcp_to_pt(self):
        """Convert distributed checkpoint to PyTorch format."""
        print("Converting distributed checkpoint to PyTorch format...")
        os.makedirs(self.trained_checkpoints_dir, exist_ok=True)
        try:
            # Run the conversion script using the existing run method
            # Convert from the temp checkpoint directory to trained checkpoints
            run(
                [
                    sys.executable,
                    "scripts/convert_distcp_to_pt.py",
                    f"{self.temp_checkpoint_dir}/model",
                    self.temp_checkpoint_dir,
                ]
            )
            print("Checkpoint conversion completed.")
            self.pipeline_state["checkpoints_converted"] = True
        finally:
            pass  # Cleanup if needed

        return self

    def run_inference(
        self,
        num_input_frames=1,
        num_output_frames=16,
        guidance=7,
        seed=0,
        context_parallel_size=1,
        input_root="assets/video2world_cosmos_nemo_assets",
    ):
        """
        Run inference with the trained model.

        Args:
            num_input_frames (int): Number of input frames
            num_output_frames (int): Number of output frames to generate
            guidance (float): Guidance scale
            seed (int): Random seed
            context_parallel_size (int): Context parallel size
            input_root (str): Input directory for inference
        """
        print("Running inference...")
        run(
            [
                sys.executable,
                "examples/inference.py",
                "-i",
                "assets/video2world_cosmos_nemo_assets/nemo_image2world.json",
                "-o",
                "outputs/cosmos_nemo_posttraining",
                "--checkpoint-path",
                f"{self.temp_checkpoint_dir}/model_ema_bf16.pt",
                "--experiment",
                "predict2_video2world_training_2b_cosmos_nemo_assets",
            ]
        )

        print("Inference completed.")
        self.pipeline_state["inference_completed"] = True
        return self

    def get_pipeline_status(self):
        """Get the current status of the pipeline."""
        print("\n=== Pipeline Status ===")
        for step, completed in self.pipeline_state.items():
            status = "✓" if completed else "✗"
            print(f"{status} {step.replace('_', ' ').title()}")
        print("=======================\n")
        return self.pipeline_state


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Post-training pipeline for Nemo assets", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--max_iters",
        type=int,
        default=1,
        help="Maximum training iterations (can be overridden by MAX_ITERS env variable)",
    )

    parser.add_argument(
        "--checkpoint_save_iter", type=int, default=None, help="Frequency to save model. Defaults to `max_iters//2`"
    )

    parser.add_argument(
        "--datasets_dir", type=str, default="datasets/cosmos_nemo_assets", help="Directory for storing datasets"
    )

    parser.add_argument(
        "--checkpoints_base_dir",
        type=str,
        default="/root/.cache/huggingface/hub",
        help="Base directory for model checkpoints",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/video2world_cosmos_nemo_posttraining",
        help="Directory for inference outputs",
    )

    parser.add_argument(
        "--temp_checkpoint_base_dir",
        type=str,
        default=("/tmp/imaginaire4-output/cosmos_predict_v2p5/video2world/2b_cosmos_nemo_assets/checkpoints"),
        help="Base temporary directory for training checkpoints",
    )

    parser.add_argument(
        "--prompt", type=str, default="A video of sks teal robot.", help="Text prompt for T5 embeddings"
    )

    parser.add_argument(
        "--experiment_name",
        type=str,
        default="predict2_video2world_training_2b_cosmos_nemo_assets",
        help="Name of the experiment configuration",
    )

    parser.add_argument("--skip_download", action="store_true", help="Skip asset download step")

    parser.add_argument("--skip_training", action="store_true", help="Skip training step")

    parser.add_argument("--skip_inference", action="store_true", help="Skip inference step")

    return parser.parse_args()


# Backward compatibility - keep original main function
def main():
    """Main function with command-line argument support."""
    args = parse_arguments()

    # Create pipeline with parsed arguments
    pipeline = PostTrainNemoAssets(
        max_iters=args.max_iters,
        checkpoint_save_iter=args.checkpoint_save_iter or max(1, args.max_iters // 2),
        datasets_dir=args.datasets_dir,
        checkpoints_base_dir=args.checkpoints_base_dir,
        output_dir=args.output_dir,
        temp_checkpoint_base_dir=args.temp_checkpoint_base_dir,
        prompt=args.prompt,
        experiment_name=args.experiment_name,
    )

    print(f"Starting pipeline with max_iters={pipeline.max_iters}")

    # Run pipeline with optional step skipping
    os.makedirs("output", exist_ok=True)

    # Always run basic setup steps
    if not args.skip_download:
        pipeline.download_assets().list_assets()
        pipeline.create_prompts_for_nemo_assets()

    # Training step
    if not args.skip_training:
        pipeline.run_training()
        pipeline.list_checkpoints()
        pipeline.convert_distcp_to_pt()  # Convert distributed checkpoint to PyTorch format

    # Inference step
    if not args.skip_inference:
        pipeline.run_inference()

    print("Pipeline completed!")
    pipeline.get_pipeline_status()


if __name__ == "__main__":
    main()
