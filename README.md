# "If I were a girl" - Magic Mirror by StarGAN

This is complementary source code for my **[blog post](https://www.dlology.com/blog/if-i-were-a-girl-magic-mirror-by-stargan/)**.

 Here is the [YouTube demo](https://youtu.be/PkWIalWnYUg).

![Magic Mirror](https://gitcdn.xyz/cdn/Tony607/blog_statics/2d525ea96d2064895160e666d805295cf97906f0/images/mirror/magic-mirror.png "Magic Mirror")



## Dependencies
- Python3.5
- PyTorch
- numpy
- opencv 1.0.1+
- opencv-python 3.3.0+contrib

Tested on:
- Windows 10 with PyTorch GPU

### install requirements
1. Install CUDA 9 from [Nvidia Developer website](https://developer.nvidia.com/cuda-90-download-archive).

2. Install [PyTorch](https://pytorch.org/) with CUDA 9 support
3. Install other Python libraries by pip.

```
pip3 install -r requirements.txt
```

## Run the demo
Plugin in a USB webcam, then run.
```
python3 DeepMagicMirror.py
```

For more instructions, check out my [tutorial](https://www.dlology.com/blog/if-i-were-a-girl-magic-mirror-by-stargan/).