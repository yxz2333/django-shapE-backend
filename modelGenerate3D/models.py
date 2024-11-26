from django.db import models


class HistoryData(models.Model):
    time = models.CharField(max_length=80)
    prompt = models.CharField(max_length=200)
    batch_size = models.IntegerField()
    guidance_scale = models.FloatField()
    render_mode = models.CharField(max_length=10)
    size = models.IntegerField()

    def __str__(self):
        return f"""
    提示词：{self.prompt}，
    生成数量：{self.batch_size}，
    提示词引导参数：{self.guidance_scale}，
    渲染模式：{self.render_mode}，
    渲染尺寸：{self.size}
    """
