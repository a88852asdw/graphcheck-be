import os
import json
import logging
from typing import Any, Dict, List, Optional, Tuple

from flask import Flask, request, jsonify, Response, stream_with_context
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from openai import AzureOpenAI
from dotenv import load_dotenv

load_dotenv()

# 基础配置
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "https://graphcheck-chatgpt.openai.azure.com/").strip()
AZURE_OPENAI_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-5.2").strip()
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview").strip()

HOST = os.getenv("HOST", "0.0.0.0")
# PORT = int(os.getenv("PORT", "8000"))
# DEBUG = os.getenv("DEBUG", "false").lower() == "true"

# 请求限制
MAX_TEXT_LENGTH = int(os.getenv("MAX_TEXT_LENGTH", "200000"))  # 单条文本最大字符数
MAX_MESSAGES_COUNT = int(os.getenv("MAX_MESSAGES_COUNT", "50"))  # messages 最大数量

# 图片/文件大小限制（字节）
MAX_IMAGE_BYTES = int(os.getenv("MAX_IMAGE_BYTES", str(5 * 1024 * 1024)))   # 10MB
MAX_FILE_BYTES = int(os.getenv("MAX_FILE_BYTES", str(10 * 1024 * 1024)))    # 20MB

# 允许的数据 URL MIME
ALLOWED_IMAGE_MIME_PREFIX = [
    "data:image/png;base64,",
    "data:image/jpeg;base64,",
    "data:image/jpg;base64,",
    "data:image/webp;base64,",
    "data:image/gif;base64,"
]

ALLOWED_FILE_MIME_PREFIX = [
    "data:application/pdf;base64,",
    "data:text/plain;base64,",
    "data:text/csv;base64,",
    "data:application/json;base64,",
    "data:application/vnd.openxmlformats-officedocument.wordprocessingml.document;base64,",
    "data:application/msword;base64,",
    "data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,",
    "data:application/vnd.ms-excel;base64,"
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Azure OpenAI 客户端
if not AZURE_OPENAI_ENDPOINT:
    raise ValueError("缺少环境变量 AZURE_OPENAI_ENDPOINT")

if not AZURE_OPENAI_DEPLOYMENT_NAME:
    raise ValueError("缺少环境变量 AZURE_OPENAI_DEPLOYMENT_NAME")

credential = DefaultAzureCredential()
token_provider = get_bearer_token_provider(
    credential,
    "https://cognitiveservices.azure.com/.default"
)

client = AzureOpenAI(
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_version=AZURE_OPENAI_API_VERSION,
    azure_ad_token_provider=token_provider
)

# 工具函数
def error_response(message: str, code: str = "BAD_REQUEST", status: int = 400):
    return jsonify({
        "success": False,
        "error": {
            "code": code,
            "message": message
        }
    }), status


def success_response(data: Dict[str, Any]):
    return jsonify({
        "success": True,
        "data": data
    })


def estimate_base64_size(data_url: str) -> int:
    """
    根据 data URL 估算 base64 解码后的字节数
    """
    try:
        if "," not in data_url:
            return 0
        _, b64data = data_url.split(",", 1)
        padding = b64data.count("=")
        return (len(b64data) * 3) // 4 - padding
    except Exception:
        return 0


def is_data_url(value: str) -> bool:
    return isinstance(value, str) and value.startswith("data:")


def validate_data_url_size(data_url: str, allowed_prefixes: List[str], max_bytes: int, field_name: str) -> Optional[str]:
    """
    校验 data URL 的 MIME 前缀和大小
    """
    if not any(data_url.startswith(prefix) for prefix in allowed_prefixes):
        return f"{field_name} 类型不被允许"

    size = estimate_base64_size(data_url)
    if size <= 0:
        return f"{field_name} 数据格式无效"

    if size > max_bytes:
        return f"{field_name} 大小超限，最大允许 {max_bytes} bytes，当前约 {size} bytes"

    return None


def validate_text_content(text: Any, field_name: str = "text") -> Optional[str]:
    if not isinstance(text, str):
        return f"{field_name} 必须是字符串"
    if len(text) > MAX_TEXT_LENGTH:
        return f"{field_name} 长度超限，最大允许 {MAX_TEXT_LENGTH} 字符"
    return None


def validate_message_item(item: Dict[str, Any], message_index: int, content_index: Optional[int] = None) -> Optional[str]:
    """
    校验 content 数组里的单个元素
    典型类型：
    - {"type": "text", "text": "..."}
    - {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}
    - {"type": "file", "file": {"filename": "...", "file_data": "data:application/pdf;base64,..."}}
    """
    if not isinstance(item, dict):
        return f"messages[{message_index}] content item 必须是对象"

    item_type = item.get("type")
    prefix = f"messages[{message_index}]"
    if content_index is not None:
        prefix += f".content[{content_index}]"

    if item_type == "text":
        err = validate_text_content(item.get("text"), f"{prefix}.text")
        if err:
            return err
        return None

    elif item_type == "image_url":
        image_url_obj = item.get("image_url")
        if not isinstance(image_url_obj, dict):
            return f"{prefix}.image_url 必须是对象"

        url = image_url_obj.get("url")
        if not isinstance(url, str) or not url:
            return f"{prefix}.image_url.url 必须是非空字符串"

        if is_data_url(url):
            err = validate_data_url_size(
                data_url=url,
                allowed_prefixes=ALLOWED_IMAGE_MIME_PREFIX,
                max_bytes=MAX_IMAGE_BYTES,
                field_name=f"{prefix}.image_url.url"
            )
            if err:
                return err

        return None

    elif item_type == "file":
        file_obj = item.get("file")
        if not isinstance(file_obj, dict):
            return f"{prefix}.file 必须是对象"

        filename = file_obj.get("filename")
        if filename is not None and not isinstance(filename, str):
            return f"{prefix}.file.filename 必须是字符串"

        file_data = file_obj.get("file_data")
        if not isinstance(file_data, str) or not file_data:
            return f"{prefix}.file.file_data 必须是非空字符串"

        if is_data_url(file_data):
            err = validate_data_url_size(
                data_url=file_data,
                allowed_prefixes=ALLOWED_FILE_MIME_PREFIX,
                max_bytes=MAX_FILE_BYTES,
                field_name=f"{prefix}.file.file_data"
            )
            if err:
                return err
        else:
            return f"{prefix}.file.file_data 目前仅支持 data URL 格式"

        return None

    else:
        return f"{prefix} 不支持的 content.type: {item_type}"


def validate_messages(messages: Any) -> Optional[str]:
    """
    支持两种 content 形态：
    1. content 为字符串
    2. content 为数组（text/image_url/file等）
    """
    if not isinstance(messages, list):
        return "messages 必须是数组"

    if len(messages) == 0:
        return "messages 不能为空"

    if len(messages) > MAX_MESSAGES_COUNT:
        return f"messages 数量超限，最大允许 {MAX_MESSAGES_COUNT}"

    allowed_roles = {"system", "user", "assistant", "tool"}

    for i, msg in enumerate(messages):
        if not isinstance(msg, dict):
            return f"messages[{i}] 必须是对象"

        role = msg.get("role")
        if role not in allowed_roles:
            return f"messages[{i}].role 非法，必须是 {sorted(list(allowed_roles))}"

        if "content" not in msg:
            return f"messages[{i}].content 缺失"

        content = msg.get("content")

        # content 为字符串
        if isinstance(content, str):
            err = validate_text_content(content, f"messages[{i}].content")
            if err:
                return err

        # content 为数组（多模态）
        elif isinstance(content, list):
            if len(content) == 0:
                return f"messages[{i}].content 不能为空数组"

            for j, item in enumerate(content):
                err = validate_message_item(item, i, j)
                if err:
                    return err

        else:
            return f"messages[{i}].content 必须是字符串或数组"

    return None


def build_request_payload(body: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    从请求体提取并校验参数
    """
    messages = body.get("messages")
    err = validate_messages(messages)
    if err:
        return None, err

    payload = {
        "model": body.get("model") or AZURE_OPENAI_DEPLOYMENT_NAME,
        "messages": messages
    }

    # 可选参数透传
    optional_fields = [
        "temperature",
        "top_p",
        "max_tokens",
        "presence_penalty",
        "frequency_penalty",
        "stop",
        "response_format",
        "tools",
        "tool_choice"
    ]

    for field in optional_fields:
        if field in body and body[field] is not None:
            payload[field] = body[field]

    return payload, None


def serialize_chat_response(resp: Any) -> Dict[str, Any]:
    """
    将 SDK 返回对象转成可 JSON 化的结构
    """
    result = {
        "id": getattr(resp, "id", None),
        "model": getattr(resp, "model", None),
        "created": getattr(resp, "created", None),
        "usage": None,
        "choices": []
    }

    usage = getattr(resp, "usage", None)
    if usage:
        result["usage"] = {
            "prompt_tokens": getattr(usage, "prompt_tokens", None),
            "completion_tokens": getattr(usage, "completion_tokens", None),
            "total_tokens": getattr(usage, "total_tokens", None)
        }

    choices = getattr(resp, "choices", []) or []
    for choice in choices:
        message = getattr(choice, "message", None)
        result["choices"].append({
            "index": getattr(choice, "index", None),
            "finish_reason": getattr(choice, "finish_reason", None),
            "message": {
                "role": getattr(message, "role", None) if message else None,
                "content": getattr(message, "content", None) if message else None
            }
        })

    return result


# 健康检查
@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "success": True,
        "message": "ok"
    })


# 普通 chat 接口
@app.route("/chat", methods=["POST"])
def chat():
    try:
        body = request.get_json(silent=True)
        if not body:
            return error_response("请求体必须是 JSON", "INVALID_JSON", 400)

        payload, err = build_request_payload(body)
        if err:
            return error_response(err, "INVALID_MESSAGES", 400)

        logger.info("chat request received")

        resp = client.chat.completions.create(**payload)

        return success_response(serialize_chat_response(resp))

    except Exception as e:
        logger.exception("chat error")
        return error_response(f"调用模型失败: {str(e)}", "CHAT_FAILED", 500)


# 流式 chat 接口
@app.route("/chat/stream", methods=["POST"])
def chat_stream():
    try:
        body = request.get_json(silent=True)
        if not body:
            return error_response("请求体必须是 JSON", "INVALID_JSON", 400)

        payload, err = build_request_payload(body)
        if err:
            return error_response(err, "INVALID_MESSAGES", 400)

        payload["stream"] = True

        logger.info("chat stream request received")

        def generate():
            try:
                stream = client.chat.completions.create(**payload)

                for chunk in stream:
                    try:
                        chunk_dict = {
                            "id": getattr(chunk, "id", None),
                            "model": getattr(chunk, "model", None),
                            "created": getattr(chunk, "created", None),
                            "choices": []
                        }

                        choices = getattr(chunk, "choices", []) or []
                        for choice in choices:
                            delta = getattr(choice, "delta", None)
                            chunk_dict["choices"].append({
                                "index": getattr(choice, "index", None),
                                "finish_reason": getattr(choice, "finish_reason", None),
                                "delta": {
                                    "role": getattr(delta, "role", None) if delta else None,
                                    "content": getattr(delta, "content", None) if delta else None
                                }
                            })

                        # SSE 格式
                        yield f"data: {json.dumps(chunk_dict, ensure_ascii=False)}\n\n"

                    except Exception as inner_e:
                        error_chunk = {
                            "error": {
                                "code": "STREAM_CHUNK_ERROR",
                                "message": str(inner_e)
                            }
                        }
                        yield f"data: {json.dumps(error_chunk, ensure_ascii=False)}\n\n"

                yield "data: [DONE]\n\n"

            except Exception as e:
                logger.exception("stream error")
                error_chunk = {
                    "error": {
                        "code": "STREAM_FAILED",
                        "message": str(e)
                    }
                }
                yield f"data: {json.dumps(error_chunk, ensure_ascii=False)}\n\n"

        return Response(
            stream_with_context(generate()),
            mimetype="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"
            }
        )

    except Exception as e:
        logger.exception("chat_stream error")
        return error_response(f"流式调用失败: {str(e)}", "STREAM_FAILED", 500)


# 启动
if __name__ == "__main__":
    # app.run(host=HOST, port=PORT, debug=DEBUG)
    app.run(host=HOST)
