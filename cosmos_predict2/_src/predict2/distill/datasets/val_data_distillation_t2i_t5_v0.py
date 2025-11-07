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

import os

from cosmos_predict2._src.imaginaire.utils.easy_io import easy_io
from cosmos_predict2._src.predict2.inference.get_t5_emb import get_text_embedding

PREFIX = "s3://bucket/cosmos_diffusion_v2/val_data/distillation_t2i_t5_v1"

prompts = [
    "an alarm clock",
    "a stop sign",
    "a robot arm",
    "A black remote control on a wooden table",
    "a tiny astronaut hatching from an egg on the moon",
    "a stunning and luxurious bedroom carved into a rocky mountainside seamlessly blending nature with modern design with a plush earth-toned bed textured stone walls circular fireplace massive uniquely shaped window framing snow-capped mountains dense forests",
    "a 3D model of a Honey badger",
    "A Casio G-Shock digital watch with a metallic silver bezel and a black face. The watch displays the time as 11:44 AM on Thursday, March 22nd, with additional features like Bluetooth connectivity, water resistance up to 20 bar, and multi-band 6 radio wave reception. The watch strap appears to be made of stainless steel, and the overall design emphasizes durability and functionality.",
    "Three vintage movie posters displayed on a brick wall, illuminated by warm golden light, showcasing different genres: sci-fi, romance, and horror, in bold, eye-catching colors.",
]

prompts = [
    "a stop sign",
    "an alarm clock",
    "a robot arm",
    "a cat holding a sign that says hello world",
    "A black remote control on a wooden table",
    "a tiny astronaut hatching from an egg on the moon",
    "a 3D model of a Honey badger",
    "In a sunny backyard, an alarm clock sits on a wooden table outside, while an ant scurries below it, searching for crumbs. Above, a ceiling fan spins lazily, casting shadows, and a flying disc soars through the air to the left, caught mid-flight against a clear blue sky.",
    "a stunning and luxurious bedroom carved into a rocky mountainside seamlessly blending nature with modern design with a plush earth-toned bed textured stone walls circular fireplace massive uniquely shaped window framing snow-capped mountains dense forests",
    "Pirate ship trapped in a cosmic maelstrom nebula, rendered in cosmic beach whirlpool engine, volumetric lighting, spectacular, ambient lights, light pollution, cinematic atmosphere, art nouveau style, illustration art artwork by SenseiJaye, intricate detail.",
    "A Casio G-Shock digital watch with a metallic silver bezel and a black face. The watch displays the time as 11:44 AM on Thursday, March 22nd, with additional features like Bluetooth connectivity, water resistance up to 20 bar, and multi-band 6 radio wave reception. The watch strap appears to be made of stainless steel, and the overall design emphasizes durability and functionality.",
    'A close-up view of a compact, ergonomic keyboard with a minimalist design. The keyboard features a combination of white and gray keycaps, with the white keys containing black characters for letters and numbers, while the gray keys have Japanese characters printed on them. The layout includes standard alphanumeric keys, function keys, and a few special keys such as "ESC," "Tab," and "Shift Lock." The keyboard has a sleek, modern aesthetic with rounded edges and a slightly raised wrist rest at the bottom. The surface it rests on appears to be a smooth, metallic or polished dark gray, reflecting some light and adding to the overall clean and professional look of the image. The focus is sharp, highlighting the texture and details of the keys and the keyboard\'s build quality.',
    "Three vintage movie posters displayed on a brick wall, illuminated by warm golden light, showcasing different genres: sci-fi, romance, and horror, in bold, eye-catching colors.",
    "A hand holds a delicate thank-you card featuring a watercolor succulent design and black cursive lettering, adorned with pearls and gold string.",
    "A group of eight individuals, including children and an adult, stand behind a table laden with various baked goods and treats. They are dressed in winter clothing, suggesting a cold environment. The table displays an assortment of desserts, including cupcakes, cookies, and other pastries, some still in their packaging. The setting appears to be outdoors, evidenced by the snow on the ground. The group seems cheerful and engaged, possibly participating in a community event or bake sale.",
    'black forest gateau cake spelling out the words "COSMOS PREDICT2", tasty, food photography, dynamic shot',
    'A poster depicting GPUs with text "Keep Calm and Buy GPUs"',
    'Three people are playing guitar in a street. There is a sign that reads "Music Fest 2025"',
]


def upload_val_data():
    easy_io.set_s3_backend(
        backend_args={
            "backend": "s3",
            "path_mapping": None,
            "s3_credential_path": "credentials/s3_checkpoint.secret",
        }
    )

    whole_content = {
        "length": len(prompts),
        "content": {},
    }
    for ith, prompt in enumerate(prompts):
        print(f"Uploading {ith} / {len(prompts)}")
        text_embedding = get_text_embedding(prompt)
        fp = f"{PREFIX}/{ith}.pt"
        whole_content["content"][ith] = {
            "prompt": prompt,
            "t5_emb_fp": fp,
        }
        easy_io.dump(text_embedding.cpu(), fp)

    easy_io.dump(whole_content, f"{PREFIX}/meta.json")
    easy_io.copyfile_from_local(__file__, f"{PREFIX}/test_upload.py")


def get_val_data(
    credential_path: str = "credentials/s3_checkpoint.secret",
):
    _key = "_s3_val_data_distillation_t2i_t5_v0"
    assert os.path.exists(credential_path), f"credential_path {credential_path} does not exist"
    easy_io.set_s3_backend(
        key=_key,
        backend_args={
            "backend": "s3",
            "path_mapping": None,
            "s3_credential_path": credential_path,
        },
    )
    kv_prompt_to_t5_emb = {}
    for ith, prompt in enumerate(prompts):
        t5_emb_fp = f"{PREFIX}/{ith}.pt"
        t5_emb = easy_io.load(t5_emb_fp, backend_key=_key)
        kv_prompt_to_t5_emb[prompt] = t5_emb
    return kv_prompt_to_t5_emb


if __name__ == "__main__":
    upload_val_data()
