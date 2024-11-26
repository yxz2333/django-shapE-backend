from django.test import Client
from django.conf import settings
import os, zipfile, django
from io import BytesIO
from django.shortcuts import HttpResponse
from rest_framework.decorators import api_view
from django.http import JsonResponse
from modelGenerate3D.models import HistoryData


# 设置环境变量，指向 Django 项目的 settings 模块
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "mybackend.settings")
django.setup()  # 初始化 Django 环境

# ALLOWED_HOSTS 定义了一个列表，包含 Django 项目允许处理的主机/域名
# 其初始包含 ['127.0.0.1', 'localhost']
# Django 测试客户端 (Client) 默认使用 testserver 作为 HTTP 主机
# 这是一个虚拟主机名，用于模拟测试环境下的请求
# 所以这里得临时添加 'testserver' 到 ALLOWED_HOSTS
settings.ALLOWED_HOSTS.append("testserver")


def _test_index():
    c = Client()
    response = c.get("/")
    assert response.status_code == 200
    print(response.content)


def _test_generate():
    c = Client()
    data = {
        "prompt": input("输入提示词"),
        "batch_size": input("输入生成模型个数"),
        "guidance_scale": 12.0,
        "render_mode": "nerf",
        "size": 32,
    }
    response = c.post("/generate/text", data)
    assert response.status_code == 200
    print("test_generate() run successfully")


@api_view(["POST"])
def generate_text(request) -> HttpResponse:
    zip_buffer = BytesIO()  # 二进制流对象

    with zipfile.ZipFile(zip_buffer, "w") as zip_file:
        for file_name in os.listdir("test/a_birthday_cupcake"):
            file_path = os.path.join("test/a_birthday_cupcake", file_name)
            if os.path.isfile(file_path) and file_name.endswith(".gif"):
                with open(file_path, "rb") as f:
                    # writestr(name, data)
                    # 将文件写入 zip 中，并用 file_name 命名
                    zip_file.writestr(file_name, f.read())

    zip_buffer.seek(0)  # 二进制文件写完，将文件指针移动到流的开始位置
    return HttpResponse(zip_buffer, status=200, content_type="application/zip")


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


## 保存历史记录
@api_view(["POST"])
def history_save(request) -> HttpResponse:
    data = request.data

    ## 从 req 里获取数据
    prompt = data.get("prompt")
    batch_size = int(data.get("batch_size"))
    guidance_scale = float(data.get("guidance_scale"))
    render_mode = data.get("render_mode")
    size = int(data.get("size"))

    ## 创建一个新对象并保存到数据库
    HistoryData(
        prompt=prompt,
        batch_size=batch_size,
        guidance_scale=guidance_scale,
        render_mode=render_mode,
        size=size,
    ).save()
    return HttpResponse("保存完毕", status=200)


if __name__ == "__main__":
    _test_index()
    _test_generate()
