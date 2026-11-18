# -*- coding: utf-8 -*-
"""
智慧植保 · 病害识别 API（EdgeOne Pages Python 云函数）
路由: /api/diagnose  (由文件路径 cloud-functions/api/diagnose.py 自动映射)

环境变量（在 EdgeOne Pages 控制台 Environment Variables 配置）：
  DASHSCOPE_API_KEY  : 百炼 API Key（sk- 或 sk-ws- 开头）
  DASHSCOPE_BASE_URL : OpenAI 兼容地址（sk-ws- Key 必须用专属地址）
  VISION_MODEL       : 模型名，默认 qwen3-vl-plus
"""

import json
import os
import re
from http.server import BaseHTTPRequestHandler

PROMPT = """你是资深农业植保专家，请识别图片中的作物病害。
只输出一行，格式为「病害名称|置信度」，置信度是 0-100 的整数。
例如：小麦条锈病|92
要求：只输出这一行，不要任何解释、礼貌用语或 Markdown 标记。
若照片不是作物病害或无法判断，只输出：无法识别|0"""


def parse_answer(text):
    s = str(text or "").strip()
    s = re.sub(r"^```(?:json)?\s*", "", s)
    s = re.sub(r"\s*```$", "", s)
    disease, confidence = "", 0
    if "|" in s:
        parts = s.split("|")
        disease = parts[0].strip()
        try:
            confidence = int(round(float(parts[1].strip())))
        except (ValueError, IndexError):
            confidence = 0
    else:
        try:
            obj = json.loads(s)
            disease = str(obj.get("disease") or obj.get("病害") or obj.get("name") or "").strip()
            confidence = int(round(float(obj.get("confidence", 0))))
        except (ValueError, TypeError):
            disease = s.split("\n")[0].strip()[:40]
    if not disease or any(k in disease for k in ("无法识别", "未知", "不能判断", "不确定")):
        confidence = 0
    return {"name": disease or "无法识别", "confidence": max(0, min(100, confidence))}


def recognize(base64_image):
    api_key = os.environ.get("DASHSCOPE_API_KEY", "")
    base_url = os.environ.get("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
    model = os.environ.get("VISION_MODEL", "qwen3-vl-plus")

    if not api_key:
        return {"success": False, "error": "服务端未配置 DASHSCOPE_API_KEY 环境变量"}
    if len(base64_image) > 8 * 1024 * 1024:
        return {"success": False, "error": "图片太大，请换一张更小的照片"}
    try:
        from openai import OpenAI
    except ImportError:
        return {"success": False, "error": "服务端缺少 openai 依赖"}

    try:
        client = OpenAI(api_key=api_key, base_url=base_url, timeout=60.0)
        resp = client.chat.completions.create(
            model=model,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": PROMPT},
                    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64_image}},
                ],
            }],
            temperature=0.01,
            max_tokens=200,
        )
        ans = parse_answer(resp.choices[0].message.content)
        return {"success": True, **ans}
    except Exception as e:
        ename = type(e).__name__
        emsg = str(e)
        if ename == "AuthenticationError" or "401" in emsg:
            err = "API Key 无效，请检查 DASHSCOPE_API_KEY"
        elif ename == "NotFoundError" or "404" in emsg or "model not exist" in emsg.lower():
            err = "模型名错误，请检查 VISION_MODEL 是否为 qwen3-vl-plus"
        elif "403" in emsg or "unpurchased" in emsg.lower() or "denied" in emsg.lower():
            err = "模型未开通，或 sk-ws- Key 未配专属 DASHSCOPE_BASE_URL"
        elif ename == "RateLimitError" or "429" in emsg:
            err = "额度用完或调用太频繁，请稍后重试"
        else:
            err = "识别服务错误：" + ename
        return {"success": False, "error": err}


class handler(BaseHTTPRequestHandler):
    def reply(self, code, obj):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.reply(200, {})

    def do_GET(self):
        self.reply(200, {"status": "ok", "hint": "POST /api/diagnose with {image: base64}"})

    def do_POST(self):
        try:
            length = int(self.headers.get("Content-Length") or 0)
            if length > 8 * 1024 * 1024:
                return self.reply(400, {"success": False, "error": "图片过大"})
            raw = self.rfile.read(length).decode("utf-8") if length else ""
            data = json.loads(raw) if raw else {}
            image = re.sub(r"^data:image/[a-zA-Z0-9.+-]+;base64,", "", str(data.get("image") or ""))
            if not image:
                return self.reply(400, {"success": False, "error": "没有收到图片数据"})
            self.reply(200, recognize(image))
        except Exception as e:
            self.reply(200, {"success": False, "error": str(e)})

    def log_message(self, fmt, *args):
        pass
