import logging
import torch

from shap_e.diffusion.sample import sample_latents
from shap_e.diffusion.gaussian_diffusion import diffusion_from_config
from shap_e.models.download import load_model, load_config
from shap_e.util.notebooks import (
    create_pan_cameras,
    decode_latent_images,
    decode_latent_mesh,
)

from io import BytesIO
import os, time, zipfile
import win32api, win32gui, win32com.client

from rest_framework.decorators import api_view
from django.shortcuts import HttpResponse
from django.http import JsonResponse
from django.conf import settings
from modelGenerate3D.models import HistoryData

PROJECT_ROOT = settings.BASE_DIR

model, device, xm, diffusion = None, None, None, None


## 初始化
@api_view(["GET"])
def index(request) -> HttpResponse:
    global device, xm, diffusion, model

    # 初始化GPU、transmitter、diffusion
    if device is None or xm is None or diffusion is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        xm = load_model("transmitter", device=device)
        diffusion = diffusion_from_config(load_config("diffusion"))

    # 初始化文生模型
    if model is None:
        model = load_model("text300M", device=device)

    return HttpResponse(" Hello, world 你已成功启动后端 ", status=200)


## 文生模型启动
@api_view(["POST"])
def generate_text(request) -> HttpResponse:
    """request
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
        gif 文件
        "gifs" : <[]> 
        
        处理代码
        status : <int>{ 200, 404, 500 }
    }
    """
    global model
    try:
        if model is None:
            raise RuntimeError("模型尚未加载")
    except Exception as e:
        logging.error(str(e))
        return HttpResponse(str(e), status=500)

    CURRENT_TIME = _get_time()  # 获取当前时间，命名用
    data = request.data

    ## 创建输出目录
    ply_dir = f"output/{CURRENT_TIME}/ply"
    obj_dir = f"output/{CURRENT_TIME}/obj"
    gif_dir = f"output/{CURRENT_TIME}/gif"
    os.makedirs(ply_dir, exist_ok=True)  # exist_ok=True 防止文件夹存在抛异常
    os.makedirs(obj_dir, exist_ok=True)
    os.makedirs(gif_dir, exist_ok=True)

    ## 从 req 里获取数据
    prompt = data.get("prompt")
    batch_size = int(data.get("batch_size"))
    guidance_scale = float(data.get("guidance_scale"))

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

    zip_buffer = BytesIO()  # 二进制流对象，存 zip 二进制流

    ## 从 req 里获取数据
    render_mode = data.get("render_mode")
    size = int(data.get("size"))

    cameras = create_pan_cameras(size, device)
    with zipfile.ZipFile(zip_buffer, "w") as zip_file:
        for i, latent in enumerate(latents):

            file_name = f"{prompt}_{i}"  # 文件名字

            ## 拍摄并获取模型 gif 动图
            images = decode_latent_images(
                xm, latent, cameras, rendering_mode=render_mode
            )
            gif_io = _images_to_gif_io(images)  # 获取 gif 二进制文件

            ## 将二进制写入 .gif
            with open(f"{gif_dir}/{file_name}.gif", "wb") as f:
                f.write(gif_io.getvalue())

            ## zip_buffer 存入 gif 二进制流文件
            with open(f"{gif_dir}/{file_name}.gif", "rb") as f:
                zip_file.writestr(f"{file_name}.gif", f.read())

            ## 将模型写入 .ply 和 .obj 文件
            t = decode_latent_mesh(xm, latent).tri_mesh()
            with open(f"{ply_dir}/{file_name}.ply", "wb") as f:
                t.write_ply(f)
            with open(f"{obj_dir}/{file_name}.obj", "w") as f:
                t.write_obj(f)

    ## 创建一个新对象并保存到数据库
    HistoryData(
        time=CURRENT_TIME,
        prompt=prompt,
        batch_size=batch_size,
        guidance_scale=guidance_scale,
        render_mode=render_mode,
        size=size,
    ).save()

    zip_buffer.seek(0)  # 二进制文件写完，将文件指针移动到流的开始位置
    return HttpResponse(zip_buffer, status=200, content_type="application/zip")


## 打开本地模型文件夹
@api_view(["GET"])
def open_folder(request) -> HttpResponse:
    path = os.path.join(PROJECT_ROOT, "output")  # 找到 output 的绝对路径

    # 打开文件夹。0：父句柄; 'open'：命令; path：文件路径; None：参数和目录; 1：显示命令
    win32api.ShellExecute(0, "open", path, None, None, 1)

    time.sleep(1)  # 等待文件夹打开

    # 获取打开的文件夹的窗口。None：类名; ""：要打开的窗口的名字
    hwnd = win32gui.FindWindow(None, "output - 文件资源管理器")
    try:
        if hwnd:
            ## 模拟用户输入以激活窗口，不然系统不让使用下面的方法
            shell = win32com.client.Dispatch(
                "WScript.Shell"
            )  # 获取 Windows Script Host 对象，它提供了一些方法来执行系统任务，如运行程序、操作文件和目录、发送按键等
            shell.SendKeys("%")  # 模拟输入

            win32gui.SetForegroundWindow(hwnd)  # 将窗口置前
        else:
            raise RuntimeError("找不到本地模型文件夹")
    except Exception as e:
        logging.error(str(e))
        return HttpResponse(str(e), status=500)

    return HttpResponse("成功打开本地模型文件夹", status=200)


## 查询历史记录区间 [l, r]
@api_view(["POST"])
def history_query(request) -> JsonResponse:
    """模型对象 (models.objects) 类型为 QS字典数组 (QuerySet[<dict>])
    例如：  <QuerySet [
            {'id': 1, 'name': 'Alice', 'value': 100},
            {'id': 2, 'name': 'Bob', 'value': 200},
            ...
            ]>
    """
    l, r = int(request.data["l"]), int(request.data["r"])
    dataArray = list(
        HistoryData.objects.all().order_by("-id")[l : r + 1].values()
    )  # 将 QuerySet 转换为字典列表
    res = {"dataArray": dataArray}  # 转成字典
    return JsonResponse(res, status=200, safe=False)


## 获取历史记录总数
@api_view(["GET"])
def history_num(request) -> HttpResponse:
    num = len(HistoryData.objects.all())
    return HttpResponse(num, status=200)


## 获取当前时间
def _get_time() -> str:
    timestamp = time.time()  # 获取当前时间戳
    local_time = time.localtime(timestamp)  # 将时间戳转换为本地时间结构
    formatted_time = time.strftime("%Y_%m_%d %H_%M_%S", local_time)  # 格式化时间
    return formatted_time


## 图片转二进制gif
def _images_to_gif_io(images) -> BytesIO:
    gif_io = BytesIO()  # 创建一个内存中的二进制流对象，用于暂时存储 GIF 数据
    images[0].save(
        gif_io,  # 将生成的 GIF 数据保存到 gif_io 对象中
        format="GIF",
        save_all=True,  # 保存所有图像帧
        append_images=images[1:],  # 将其余图像列表的图像帧加到这个 GIF 文件中
        duration=100,  # 每帧的显示时间为 100 ms
        loop=0,  # GIF 动画无限循环
    )
    gif_io.seek(0)  # 二进制文件写完，将文件指针移动到流的开始位置
    return gif_io
