from django.test import Client
from django.conf import settings
import os
import django

# 设置环境变量，指向 Django 项目的 settings 模块
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'mybackend.settings')
django.setup()  # 初始化 Django 环境

# ALLOWED_HOSTS 定义了一个列表，包含 Django 项目允许处理的主机/域名
# 其初始包含 ['127.0.0.1', 'localhost']
# Django 测试客户端 (Client) 默认使用 testserver 作为 HTTP 主机
# 这是一个虚拟主机名，用于模拟测试环境下的请求
# 所以这里得临时添加 'testserver' 到 ALLOWED_HOSTS
settings.ALLOWED_HOSTS.append('testserver')


def test_index():
    c = Client()
    response = c.get('/')
    assert response.status_code == 200
    print(response.content)


def test_image():
    c = Client()
    response = c.get('/model/text')
    assert response.status_code == 200
    print("test_image() run successfully")


def test_generate():
    c = Client()
    data = {
        "prompt": input("输入提示词"),
        "batch_size": input("输入生成模型个数"),
        "guidance_scale": 12.0,
        "render_mode": 'nerf',
        "size": 32
    }
    response = c.post('/generate/text', data)
    assert response.status_code == 200
    print("test_generate() run successfully")


if __name__ == '__main__':
    test_index()
    test_image()
    test_generate()
    