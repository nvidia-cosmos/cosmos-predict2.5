<p align="center">
    <img src="https://github.com/user-attachments/assets/28f2d612-bbd6-44a3-8795-833d05e9f05f" width="274" alt="NVIDIA Cosmos"/>
</p>

<p align="center">
  <a href="https://www.nvidia.com/en-us/ai/cosmos">Product Website</a>&nbsp | 🤗 <a href="https://huggingface.co/collections/nvidia/cosmos-predict25-68bb63255f2fc206c5e5b346">Hugging Face</a>&nbsp | <a href="https://research.nvidia.com/publication/2025-09_world-simulation-video-foundation-models-physical-ai">Paper</a>&nbsp | <a href="https://research.nvidia.com/labs/dir/cosmos-predict2.5">Paper Website</a>
</p>

NVIDIA Cosmos™ is a platform purpose-built for physical AI, featuring state-of-the-art generative world foundation models (WFMs), robust guardrails, and an accelerated data processing and curation pipeline. Designed specifically for real-world systems, Cosmos enables developers to rapidly advance physical AI applications such as autonomous vehicles (AVs), robots, and video analytics AI agents.

Cosmos World Foundation Models come in three model types which can all be customized in post-training: [cosmos-predict](https://github.com/nvidia-cosmos/cosmos-predict2.5), [cosmos-transfer](https://github.com/nvidia-cosmos/cosmos-transfer2.5), and [cosmos-reason](https://github.com/nvidia-cosmos/cosmos-reason1).

## News!
* [October 14, 2025] We released the Cosmos-Predict2.5 robot/action-cond: [Inference Guide](docs/inference_robot_action_cond.md) and [Post-Training Guide](docs/post-training_video2world_action.md). Also released [Auto Multview Post-Training](docs/post-training_multiview.md).
* [October 6, 2025] We released [Cosmos-Predict2.5](https://github.com/nvidia-cosmos/cosmos-predict2.5) and [Cosmos-Transfer2.5](https://github.com/nvidia-cosmos/cosmos-transfer2.5) - the next generation of our world simulation models!

## Cosmos-Predict2.5

We introduce Cosmos-Predict2.5, the latest version of the Cosmos World Foundation Models (WFMs) family, specialized for simulating and predicting the future state of the world in the form of video. Cosmos-Predict2.5 is a flow based model that unifies Text2World, Image2World, and Video2World into a single model and utilizes Cosmos-Reason1, a Physical AI reasoning vision language model (VLM), as the text encoder. Cosmos-Predict2.5 significantly improves upon Cosmos-Predict1 in both quality and prompt alignment.

### Image2World

<details><summary>Input prompt</summary>
A nighttime city bus terminal gradually shifts from stillness to subtle movement. At first, multiple double-decker buses are parked under the glow of overhead lights, with a central bus labeled '87D' facing forward and stationary. As the video progresses, the bus in the middle moves ahead slowly, its headlights brightening the surrounding area and casting reflections onto adjacent vehicles. The motion creates space in the lineup, signaling activity within the otherwise quiet station. It then comes to a smooth stop, resuming its position in line. Overhead signage in Chinese characters remains illuminated, enhancing the vibrant, urban night scene.
</details>

| Input image | Output video
| --- | --- |
| <img src="https://github.com/user-attachments/assets/c855f468-0577-475d-a2bb-5673b9d8ae91" width="500" alt="Input image" > | <video src="https://github.com/user-attachments/assets/a233567b-9eb4-405a-ab36-c0bf902d2988" width="500" alt="Output video" controls></video> |

### Video2World

<details><summary>Input prompt</summary>
A robotic arm, primarily white with black joints and cables, is shown in a clean, modern indoor setting with a white tabletop. The arm, equipped with a gripper holding a small, light green pitcher, is positioned above a clear glass containing a reddish-brown liquid and a spoon. The robotic arm is in the process of pouring a transparent liquid into the glass. To the left of the pitcher, there is an opened jar with a similar reddish-brown substance visible through its transparent body. In the background, a vase with white flowers and a brown couch are partially visible, adding to the contemporary ambiance. The lighting is bright, casting soft shadows on the table. The robotic arm's movements are smooth and controlled, demonstrating precision in its task. As the video progresses, the robotic arm completes the pour, leaving the glass half-filled with the reddish-brown liquid. The jar remains untouched throughout the sequence, and the spoon inside the glass remains stationary. The other robotic arm on the right side also stays stationary throughout the video. The final frame captures the robotic arm with the pitcher finishing the pour, with the glass now filled to a higher level, while the pitcher is slightly tilted but still held securely by the gripper.
</details>

| Input Video | Output Video
| --- | --- |
| <video src="https://github.com/user-attachments/assets/ddca366e-b30f-44bb-9def-b4a8386d8d23" width="500" alt="Output video" controls></video> | <video src="https://github.com/user-attachments/assets/62c0800d-036a-4dbc-b0a6-199ee25d8e31" width="500" alt="Output video" controls></video> |

## Cosmos-Predict2.5 Model Family

Our world simulation models, Cosmos-Predict's fundamental capability is predicting future world states in video form supporting multimodal inputs. We have open sourced both pre-trained foundation models as well as post-trained models accelerating multiple domains. Please check back as we continue to add more specialized models and capabilities to the Predict family!

[**Cosmos-Predict2.5**](docs/inference.md): Base [checkpoints](https://huggingface.co/nvidia/Cosmos-Predict2.5-2B/tree/main/base), trained from the ground up for Physical AI and robotics.

[**Cosmos-Predict2.5/auto/multiview**](docs/inference_auto_multiview.md): Specialized [checkpoints](https://huggingface.co/nvidia/Cosmos-Predict2.5-2B/tree/main/auto/multiview), post-trained for Autonomous Vehicle applications.

| Model Name | Capability | Input |
| --- | --- | --- |
| [**Cosmos-Predict2.5 base**](docs/inference.md) | | |
| Cosmos-Predict2.5-2B/pre-trained | pre-trained base | text + image or video |
| Cosmos-Predict2.5-2B/post-trained | post-trained base | text + image or video |
| [**Cosmos-Predict2.5 auto**](docs/inference_auto_multiview.md) | | |
| Cosmos-Predict2.5-2B/auto/multiview | driving, 7-camera view | text + image or video |
| [**Cosmos-Predict2.5-2B robot**](docs/inference_robot_action_cond.md) | | |
| Cosmos-Predict2.5-2B/robot/action-cond | robotic, action-conditioned | action |

## User Guide

* [Setup Guide](docs/setup.md)
* [Inference](docs/inference.md)
  * [Auto Multiview](docs/inference_auto_multiview.md)
  * [Robot Action-Conditiond](docs/inference_robot_action_cond.md)
* [Post-Training](docs/post-training.md)
  * [Video2World Cosmos-NeMo-Assets](docs/post-training_video2world_cosmos_nemo_assets.md)
  * [Video2World DreamGen Bench](docs/post-training_video2world_gr00t.md)
  * [Auto Multiview](docs/post-training_multiview.md)
  * [Robot Action-Conditioned](docs/post-training_video2world_action.md)
* [Troubleshooting](docs/troubleshooting.md)

## Contributing

We thrive on community collaboration! [NVIDIA-Cosmos](https://github.com/nvidia-cosmos/) wouldn't be where it is without contributions from developers like you. Check out our [Contributing Guide](CONTRIBUTING.md) to get started, and share your feedback through issues.

Big thanks 🙏 to everyone helping us push the boundaries of open-source physical AI!

## License and Contact

This project will download and install additional third-party open source software projects. Review the license terms of these open source projects before use.

NVIDIA Cosmos source code is released under the [Apache 2 License](https://www.apache.org/licenses/LICENSE-2.0).

NVIDIA Cosmos models are released under the [NVIDIA Open Model License](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license). For a custom license, please contact [cosmos-license@nvidia.com](mailto:cosmos-license@nvidia.com).
