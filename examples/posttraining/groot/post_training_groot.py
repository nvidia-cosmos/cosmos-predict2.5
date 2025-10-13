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


class PostTrainGroot:
    """
    A class to manage the complete post-training pipeline for GR1 dataset.

    This class provides a structured way to handle the entire workflow from
    downloading the dataset to running inference, with configurable parameters
    and state tracking.
    """

    def __init__(
        self,
        max_iters=None,
        checkpoint_save_iter=None,
        datasets_dir="datasets/benchmark_train/gr1",
        checkpoints_base_dir="/root/.cache/huggingface/hub",
        output_dir="outputs/gr00t_gr1_posttraining",
        temp_checkpoint_base_dir=None,
        experiment_name="predict2_video2world_training_2b_groot_gr1_480",
    ):
        """
        Initialize the PostTrainGroot pipeline.

        Args:
            max_iters (int): Maximum training iterations (defaults to env MAX_ITERS or 100)
            checkpoint_save_iter (int): Frequency to save checkpoint
            datasets_dir (str): Directory for storing datasets
            checkpoints_base_dir (str): Base directory for model checkpoints
            output_dir (str): Directory for inference outputs
            temp_checkpoint_base_dir (str): Base temporary directory for training checkpoints
            experiment_name (str): Name of the experiment configuration
        """
        # Environment variable MAX_ITERS takes precedence over command line argument
        self.max_iters = int(os.getenv("MAX_ITERS", max_iters or 100))
        self.checkpoint_save_iter = checkpoint_save_iter
        self.datasets_dir = datasets_dir
        self.checkpoints_base_dir = checkpoints_base_dir
        self.output_dir = output_dir
        self.experiment_name = experiment_name

        # Use IMAGINAIRE_OUTPUT_ROOT if set, otherwise use default
        imaginaire_output_root = os.getenv("IMAGINAIRE_OUTPUT_ROOT", "/tmp/imaginaire4-output")
        if temp_checkpoint_base_dir is None:
            self.temp_checkpoint_base_dir = (
                f"{imaginaire_output_root}/cosmos_predict_v2p5/video2world/2b_groot_gr1_480/checkpoints"
            )
        else:
            self.temp_checkpoint_base_dir = temp_checkpoint_base_dir

        # Derived paths
        self.hf_download_dir = "datasets/benchmark_train/hf_gr1"
        self.videos_dir = f"{self.datasets_dir}/videos"
        self.metas_dir = f"{self.datasets_dir}/metas"
        self.trained_checkpoints_dir = f"{self.checkpoints_base_dir}/trained"
        self.base_checkpoints_dir = f"{self.checkpoints_base_dir}/models--nvidia--Cosmos-Predict2.5-2B/snapshots/a64a214a5ff6937c35ac32a41a7922442ccdf774/d20b7120-df3e-4911-919d-db6e08bad31c"

        # Build temp checkpoint dir with proper 9-digit formatting
        self.temp_checkpoint_dir = f"{self.temp_checkpoint_base_dir}/iter_{self.max_iters:09d}"

        # Pipeline state tracking
        self.pipeline_state = {
            "dataset_downloaded": False,
            "dataset_organized": False,
            "prompts_created": False,
            "training_completed": False,
            "checkpoints_converted": False,
            "inference_completed": False,
        }

        print(f"PostTrainGroot initialized with max_iters={self.max_iters}")
        print(f"Datasets dir: {self.datasets_dir}")
        print(f"Checkpoints dir: {self.checkpoints_base_dir}")
        print(f"Temp checkpoint dir: {self.temp_checkpoint_dir}")

    def download_dataset(self):
        """Download GR1-100 dataset from HuggingFace."""
        if os.path.exists(self.datasets_dir) and os.path.exists(self.videos_dir):
            print(f"Dataset already exists at {self.datasets_dir}")
            self.pipeline_state["dataset_downloaded"] = True
            self.pipeline_state["dataset_organized"] = True
            return self

        print("Downloading GR1-100 dataset...")
        run(
            [
                "hf",
                "download",
                "nvidia/GR1-100",
                "--repo-type",
                "dataset",
                "--local-dir",
                self.hf_download_dir,
            ]
        )
        print("Dataset downloaded.")
        self.pipeline_state["dataset_downloaded"] = True
        return self

    def organize_dataset(self):
        """Organize downloaded dataset into the expected structure."""
        if self.pipeline_state["dataset_organized"]:
            print("Dataset already organized.")
            return self

        print("Organizing dataset...")

        # Create necessary directories
        os.makedirs(self.videos_dir, exist_ok=True)

        # Move video files
        source_videos = f"{self.hf_download_dir}/gr1"
        if os.path.exists(source_videos):
            run(["sh", "-c", f"mv {source_videos}/*.mp4 {self.videos_dir}/"], shell=False)
            print(f"Moved videos to {self.videos_dir}")

        # Move metadata.csv
        metadata_source = f"{self.hf_download_dir}/metadata.csv"
        metadata_dest = f"{self.datasets_dir}/metadata.csv"
        if os.path.exists(metadata_source):
            shutil.move(metadata_source, metadata_dest)
            print(f"Moved metadata.csv to {self.datasets_dir}")

        print("Dataset organized.")
        self.pipeline_state["dataset_organized"] = True
        return self

    def list_dataset(self):
        """List contents of the dataset directory."""
        print("Listing dataset directory:")
        run(["ls", "-l", self.datasets_dir])
        if os.path.exists(self.videos_dir):
            print("\nListing videos directory:")
            run(["ls", "-l", self.videos_dir])
        return self

    def create_prompts(self):
        """Create prompt files for the GR1 dataset."""
        if self.pipeline_state["prompts_created"]:
            print("Prompts already created.")
            return self

        print("Creating prompts for GR1 dataset...")
        run(
            [
                sys.executable,
                "-m",
                "scripts.create_prompts_for_gr1_dataset",
                "--dataset_path",
                self.datasets_dir,
            ]
        )
        self.pipeline_state["prompts_created"] = True
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
        config = "--config=cosmos_predict2/_src/predict2/configs/video2world/config.py"

        # Build command parts for better readability
        cmd_parts = [
            f"{base_cmd} {script} {config} --",
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
        input_json="assets/sample_gr00t_dreams_gr1/gr00t_image2world.json",
    ):
        """
        Run inference with the trained model.

        Args:
            input_json (str): Path to the input JSON configuration file
        """
        print("Running inference...")
        run(
            [
                "torchrun",
                "--nproc_per_node=8",
                "examples/inference.py",
                "-i",
                input_json,
                "-o",
                self.output_dir,
                "--checkpoint-path",
                f"{self.temp_checkpoint_dir}/model_ema_bf16.pt",
                "--experiment",
                self.experiment_name,
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
        description="Post-training pipeline for GR1 dataset", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--max_iters",
        type=int,
        default=2,
        help="Maximum training iterations (can be overridden by MAX_ITERS env variable)",
    )

    parser.add_argument(
        "--checkpoint_save_iter", type=int, default=None, help="Frequency to save model. Defaults to `max_iters//2`"
    )

    parser.add_argument(
        "--datasets_dir", type=str, default="datasets/benchmark_train/gr1", help="Directory for storing datasets"
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
        default="outputs/gr00t_gr1_posttraining",
        help="Directory for inference outputs",
    )

    parser.add_argument(
        "--temp_checkpoint_base_dir",
        type=str,
        default=None,
        help="Base temporary directory for training checkpoints (defaults to $IMAGINAIRE_OUTPUT_ROOT/cosmos_predict_v2p5/video2world/2b_groot_gr1_480/checkpoints)",
    )

    parser.add_argument(
        "--experiment_name",
        type=str,
        default="predict2_video2world_training_2b_groot_gr1_480",
        help="Name of the experiment configuration",
    )

    parser.add_argument("--skip_download", action="store_true", help="Skip dataset download step")

    parser.add_argument("--skip_training", action="store_true", help="Skip training step")

    parser.add_argument("--skip_inference", action="store_true", help="Skip inference step")

    return parser.parse_args()


# Main function with command-line argument support
def main():
    """Main function with command-line argument support."""
    args = parse_arguments()

    # Create pipeline with parsed arguments
    pipeline = PostTrainGroot(
        max_iters=args.max_iters,
        checkpoint_save_iter=args.checkpoint_save_iter or max(1, args.max_iters // 2),
        datasets_dir=args.datasets_dir,
        checkpoints_base_dir=args.checkpoints_base_dir,
        output_dir=args.output_dir,
        temp_checkpoint_base_dir=args.temp_checkpoint_base_dir,
        experiment_name=args.experiment_name,
    )

    print(f"Starting pipeline with max_iters={pipeline.max_iters}")

    # Run pipeline with optional step skipping
    os.makedirs("output", exist_ok=True)

    # Download and prepare dataset
    if not args.skip_download:
        pipeline.download_dataset()
        pipeline.organize_dataset()
        pipeline.list_dataset()
        pipeline.create_prompts()

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
