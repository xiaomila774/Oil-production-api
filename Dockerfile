# ① 选择轻量级官方 Python 镜像
FROM python:3.11-slim

# ② 把容器里的工作目录定为 /app
WORKDIR /app

# ③ 把当前文件夹所有内容复制进镜像
COPY . .

# ④ 安装依赖
RUN pip install --no-cache-dir -r requirements.txt

# ⑤ 容器暴露 8000 端口（给云平台识别用）
EXPOSE 8000

# ⑥ 容器启动时执行 uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
