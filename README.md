# Stable Diffusion Inpainting Cog model

[![Replicate](https://replicate.com/andreasjansson/stable-diffusion-inpainting/badge)](https://replicate.com/andreasjansson/stable-diffusion-inpainting)

This is an implementation of the [Stable Diffusion Inpainting](https://huggingface.co/runwayml/stable-diffusion-v1-5) as a Cog model. [Cog packages machine learning models as standard containers.](https://github.com/replicate/cog)

First, download the pre-trained weights [with your Hugging Face auth token](https://huggingface.co/settings/tokens):

    cog run script/download-weights <your-hugging-face-auth-token>

Then, you can run predictions:

    cog predict -i prompt="a herd of grazing sheep" -i image=@desktop.png -i mask=@desktop-mask.png

To push a new version you need to be @andreasjansson. If you are, then run

```
./test_and_push.sh
```
