import torch

from rest_framework.decorators import api_view

from shap_e.diffusion.sample import sample_latents
from shap_e.diffusion.gaussian_diffusion import diffusion_from_config
from shap_e.models.download import load_model, load_config
from shap_e.util.notebooks import create_pan_cameras, decode_latent_images, decode_latent_mesh

import io

from django.shortcuts import HttpResponse

model, device, xm, diffusion = None, None, None, None


@api_view(['GET'])
def index(request) -> HttpResponse:
    global device, xm, diffusion
    if device is None or xm is None or diffusion is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print()
        xm = load_model('transmitter', device=device)
        diffusion = diffusion_from_config(load_config('diffusion'))
    return HttpResponse(" Hello, world 你已成功启动后端 ", status=200)


def _images_to_gif_io(images) -> io.BytesIO:
    gif_io = io.BytesIO()  # 创建一个内存中的二进制流对象，用于暂时存储 GIF 数据
    images[0].save(
        gif_io,  # 将生成的 GIF 数据保存到 gif_io 对象中
        format="GIF",
        save_all=True,  # 保存所有图像帧
        append_images=images[1:],  # 将其余图像列表的图像帧加到这个 GIF 文件中
        duration=100,  # 每帧的显示时间为 100 ms
        loop=0  # GIF 动画无限循环
    )
    gif_io.seek(0)  # 二进制文件写完，将文件指针移动到流的开始位置
    return gif_io


@api_view(['GET'])
def text(request) -> HttpResponse:
    global model
    model = load_model("text300M", device=device)
    return HttpResponse(status=200)


@api_view(['GET'])
def image(request) -> HttpResponse:
    global model
    model = load_model('image300M', device=device)
    return HttpResponse(status=200)


@api_view(['POST'])
def generate_text(request) -> HttpResponse:
    """ request
    {
        提示词、生成个数、提示词引导参数
        "prompt" : <str>
        "batch_size" : <int>
        "guidance_scale" : <float>

        渲染模式、渲染尺寸
        "render_mode" : <str>{'nerf','stf'}
        "size" : <int>
    }
    """

    """ response
    {
        gif 二进制文件
        "gifs" : <[]io.BytesIO>
        
        处理代码
        status : <int>{ 200, 404, 500 }
        
        错误处理
        "error" : <str>
    }
    """

    global model
    try:
        if model is None:
            raise RuntimeError("Model not loaded.")
    except Exception as e:
        return HttpResponse({"error": str(e)}, status=500)

    # 从 req 里获取数据
    prompt = request.POST.get('prompt')
    batch_size = int(request.POST.get('batch_size'))
    guidance_scale = float(request.POST.get('guidance_scale'))

    latents = sample_latents(
        batch_size=batch_size,
        model=model,
        diffusion=diffusion,
        guidance_scale=guidance_scale,
        model_kwargs=dict(texts=[prompt] * batch_size),
        progress=True,
        clip_denoised=True,
        use_fp16=True,
        use_karras=True,
        karras_steps=64,
        sigma_min=1e-3,
        sigma_max=160,
        s_churn=0,
    )

    res = {"gif": []}  # 存 gif 动图

    # 从 req 里获取数据
    render_mode = request.POST.get('render_mode')
    size = int(request.POST.get('size'))

    cameras = create_pan_cameras(size, device)
    for i, latent in enumerate(latents):
        # 拍摄并获取模型 gif 动图
        images = decode_latent_images(xm, latent, cameras, rendering_mode=render_mode)
        gif = _images_to_gif_io(images)
        res['gif'].append(gif)

        # 将模型写入 .ply 和 .obj 文件
        t = decode_latent_mesh(xm, latent).tri_mesh()
        with open(f'output/ply/{prompt}_{i}.ply', 'wb') as f:
            t.write_ply(f)
        with open(f'output/obj/{prompt}_{i}.obj', 'w') as f:
            t.write_obj(f)

    return HttpResponse(res, status=200)
