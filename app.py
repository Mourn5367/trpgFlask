from flask import Flask
from flask_sock import Sock
import json
import requests
import traceback
import base64
from PIL import Image
import io

app = Flask(__name__)
sock = Sock(app)

# API 엔드포인트 설정
OLLAMA_API_URL = "http://192.168.24.189:11434/api/generate"
SD_API_URL = "http://192.168.24.189:7860/sdapi/v1"
TTS_API_URL = "http://192.168.24.189:9880/tts"

# 기본 설정
DEFAULT_MODEL = "gemma3:4b"


@app.route('/')
def index():
    return """
    <h1>통합 AI 서버</h1>
    <p>WebSocket 서버가 실행 중입니다.</p>
    <ul>
        <li>채팅: Ollama API 연동</li>
        <li>이미지 생성: Stable Diffusion API 연동</li>
        <li>음성 생성: TTS API 연동</li>
        <li>WebSocket 엔드포인트: /ws</li>
    </ul>
    """


@sock.route('/ws')
def websocket(ws):
    print("새로운 WebSocket 연결이 열렸습니다.")
    try:
        while True:
            print("메시지 대기 중...")
            data = ws.receive()
            print(f"받은 데이터: {data}")

            try:
                json_data = json.loads(data)
                print(f"파싱된 JSON: {json_data}")

                msg_type = json_data.get("type", "")
                print(f"메시지 타입: {msg_type}")

                # 채팅 관련 처리 (Ollama)
                if msg_type in ["chat", "message", "greeting"] or "message" in json_data:
                    handle_chat_message(ws, json_data)

                # 이미지 생성 관련 처리 (Stable Diffusion)
                elif msg_type == "generate_image":
                    handle_image_generation(ws, json_data)

                # 음성 생성 관련 처리 (TTS)
                elif msg_type == "generate_tts":
                    handle_tts_generation(ws, json_data)

                # 모델 목록 조회 (Stable Diffusion)
                elif msg_type == "get_models":
                    handle_get_models(ws, json_data)

                # 모델 변경 (Stable Diffusion)
                elif msg_type == "change_model":
                    handle_change_model(ws, json_data)

                # 서버 상태 확인
                elif msg_type == "ping":
                    ws.send(json.dumps({
                        "status": "ok",
                        "type": "pong",
                        "message": "서버가 정상 동작 중입니다.",
                        "services": {
                            "ollama": check_ollama_connection(),
                            "stable_diffusion": check_sd_connection(),
                            "tts": check_tts_connection()
                        }
                    }))

                # 이전 버전 호환성 (prompt만 있는 경우)
                elif "prompt" in json_data and not msg_type:
                    # 프롬프트만 있으면 이미지 생성으로 처리
                    json_data["type"] = "generate_image"
                    handle_image_generation(ws, json_data)

                else:
                    # 알 수 없는 메시지 타입
                    ws.send(json.dumps({
                        "status": "error",
                        "message": f"지원하지 않는 메시지 타입: {msg_type}",
                        "supported_types": [
                            "chat", "message", "greeting",
                            "generate_image", "generate_tts", 
                            "get_models", "change_model", "ping"
                        ],
                        "original": json_data
                    }))

            except json.JSONDecodeError as e:
                print(f"JSON 파싱 오류: {e}")
                ws.send(json.dumps({
                    "status": "error",
                    "error": "Invalid JSON",
                    "details": str(e)
                }))
            except Exception as e:
                print(f"처리 중 예외 발생: {e}")
                print(traceback.format_exc())
                ws.send(json.dumps({
                    "status": "error",
                    "error": str(e),
                    "traceback": traceback.format_exc()
                }))

    except Exception as e:
        print(f"WebSocket 연결 오류: {e}")
        print(traceback.format_exc())


def handle_chat_message(ws, json_data):
    """채팅 메시지 처리 (Ollama API)"""
    message = json_data.get("message", "")
    model = json_data.get("model", DEFAULT_MODEL)

    print(f"채팅 처리: '{message}', 모델: {model}")

    if not message.strip():
        ws.send(json.dumps({
            "status": "error",
            "error": "메시지가 비어있습니다.",
            "original": json_data
        }))
        return

    # Ollama 스트리밍 호출
    print("Ollama API 스트리밍 호출 시작...")
    stream_ollama_to_websocket(message, model, ws, json_data)
    print("Ollama API 스트리밍 호출 완료")


def handle_image_generation(ws, json_data):
    """이미지 생성 처리 (Stable Diffusion API)"""
    prompt = json_data.get("prompt", "")

    if not prompt:
        ws.send(json.dumps({
            "status": "error",
            "message": "프롬프트가 필요합니다."
        }))
        return

    # 진행 상태 메시지 전송
    ws.send(json.dumps({
        "status": "processing",
        "message": "이미지 생성 중...",
        "type": "image_generation_progress"
    }))

    try:
        params = json_data.get("params", {})
        image_data = generate_image(prompt, params)

        ws.send(json.dumps({
            "status": "success",
            "type": "image_generated",
            "image": image_data,
            "prompt": prompt,
            "original": json_data
        }))

    except Exception as e:
        print(f"이미지 생성 오류: {e}")
        ws.send(json.dumps({
            "status": "error",
            "message": str(e),
            "type": "image_generation_error",
            "original": json_data
        }))


def handle_tts_generation(ws, json_data):
    """TTS 음성 생성 처리"""
    text = json_data.get("text", "")

    if not text.strip():
        ws.send(json.dumps({
            "status": "error",
            "message": "변환할 텍스트가 필요합니다.",
            "type": "tts_error",
            "original": json_data
        }))
        return

    # 진행 상태 메시지 전송
    ws.send(json.dumps({
        "status": "processing",
        "message": "음성 생성 중...",
        "type": "tts_generation_progress"
    }))

    try:
        params = json_data.get("params", {})
        audio_data = generate_tts(text, params)

        ws.send(json.dumps({
            "status": "success",
            "type": "tts_generated",
            "audio": audio_data,
            "text": text,
            "original": json_data
        }))

    except Exception as e:
        print(f"TTS 생성 오류: {e}")
        ws.send(json.dumps({
            "status": "error",
            "message": str(e),
            "type": "tts_generation_error",
            "original": json_data
        }))


def handle_get_models(ws, json_data):
    """모델 목록 조회 처리"""
    try:
        models = get_sd_models()
        ws.send(json.dumps({
            "status": "success",
            "type": "models_list",
            "models": models,
            "original": json_data
        }))
    except Exception as e:
        ws.send(json.dumps({
            "status": "error",
            "message": str(e),
            "type": "models_error",
            "original": json_data
        }))


def handle_change_model(ws, json_data):
    """모델 변경 처리"""
    model_name = json_data.get("model_name")

    if not model_name:
        ws.send(json.dumps({
            "status": "error",
            "message": "모델 이름이 필요합니다.",
            "original": json_data
        }))
        return

    try:
        change_sd_model(model_name)
        ws.send(json.dumps({
            "status": "success",
            "type": "model_changed",
            "message": f"모델이 '{model_name}'(으)로 변경되었습니다.",
            "model_name": model_name,
            "original": json_data
        }))
    except Exception as e:
        ws.send(json.dumps({
            "status": "error",
            "message": str(e),
            "type": "model_change_error",
            "original": json_data
        }))


def stream_ollama_to_websocket(message, model, ws, original_request):
    """Ollama API를 스트리밍 모드로 호출하고 웹소켓으로 전송"""
    payload = {
        "model": model,
        "prompt": message,
        "stream": True
    }

    print(f"Ollama API 요청: {payload}")

    try:
        response = requests.post(OLLAMA_API_URL, json=payload, stream=True)
        print(f"Ollama API 응답 상태 코드: {response.status_code}")
        response.raise_for_status()

        full_response = ""
        chunk_count = 0

        print("스트리밍 응답 처리 시작...")
        for line in response.iter_lines():
            if line:
                chunk_count += 1
                print(f"청크 {chunk_count}: {line}")

                try:
                    chunk_data = json.loads(line.decode('utf-8'))

                    if 'response' in chunk_data:
                        chunk_text = chunk_data['response']
                        full_response += chunk_text

                        ws.send(json.dumps({
                            "status": "streaming",
                            "type": "chat_chunk",
                            "original": original_request,
                            "chunk": chunk_text,
                            "response_so_far": full_response,
                            "done": False,
                            "chunk_number": chunk_count
                        }))

                    if chunk_data.get('done', False):
                        print("스트리밍 완료됨")
                        ws.send(json.dumps({
                            "status": "complete",
                            "type": "chat_complete",
                            "original": original_request,
                            "response": full_response,
                            "done": True,
                            "model": model,
                            "total_duration": chunk_data.get("total_duration", 0),
                            "load_duration": chunk_data.get("load_duration", 0),
                            "prompt_eval_count": chunk_data.get("prompt_eval_count", 0),
                            "eval_count": chunk_data.get("eval_count", 0),
                            "total_chunks": chunk_count
                        }))
                        break

                except json.JSONDecodeError:
                    print(f"JSON 파싱 오류 (청크): {line}")
                    continue

        print(f"총 {chunk_count}개 청크 처리 완료")

    except requests.exceptions.RequestException as e:
        error_msg = f"Ollama API 호출 오류: {str(e)}"
        print(error_msg)
        ws.send(json.dumps({
            "status": "error",
            "type": "chat_error",
            "error": error_msg,
            "original": original_request,
            "traceback": traceback.format_exc()
        }))


def generate_image(prompt, params=None):
    """Stable Diffusion 이미지 생성"""
    if params is None:
        params = {}

    payload = {
        "prompt": prompt,
        "negative_prompt": params.get("negative_prompt", "ugly, blurry, poor quality, deformed"),
        "width": params.get("width", 512),
        "height": params.get("height", 512),
        "steps": params.get("steps", 25),
        "cfg_scale": params.get("cfg_scale", 7.0),
        "sampler_name": params.get("sampler_name", "DPM++ 2M Karras"),
        "seed": params.get("seed", -1)
    }

    print(f"Stable Diffusion 요청: {payload}")

    response = requests.post(f"{SD_API_URL}/txt2img", json=payload)

    if response.status_code != 200:
        raise Exception(f"Stable Diffusion API 오류: {response.status_code} - {response.text}")

    result = response.json()

    if "images" in result and len(result["images"]) > 0:
        return result["images"][0]
    else:
        raise Exception("이미지 생성 실패: 이미지가 반환되지 않았습니다.")


def generate_tts(text, params=None):
    """TTS 음성 생성"""
    if params is None:
        params = {}

    # 기본값 설정
    payload = {
        "text": text,
        "text_lang": params.get("text_lang", "ko"),
        "ref_audio_path": params.get("ref_audio_path", "A-A3-E-055-0101.wav"),
        "prompt_lang": params.get("prompt_lang", "ko"),
        "prompt_text": params.get("prompt_text", "지금이 범인을 찾을 땐가요, 아버지라면 당연히 생사를 오가는 딸 곁에 있어 주셔야죠!"),
        "media_type": params.get("media_type", "wav")
    }

    print(f"TTS API 요청: {payload}")

    response = requests.post(TTS_API_URL, json=payload)

    if response.status_code != 200:
        raise Exception(f"TTS API 오류: {response.status_code} - {response.text}")

    # 오디오 데이터를 base64로 인코딩하여 반환
    audio_base64 = base64.b64encode(response.content).decode('utf-8')
    return audio_base64


def get_sd_models():
    """Stable Diffusion 모델 목록 가져오기"""
    response = requests.get(f"{SD_API_URL}/sd-models")
    if response.status_code != 200:
        raise Exception(f"모델 목록 요청 실패: {response.status_code} - {response.text}")
    return response.json()


def change_sd_model(model_name):
    """Stable Diffusion 모델 변경"""
    payload = {"sd_model_checkpoint": model_name}
    response = requests.post(f"{SD_API_URL}/options", json=payload)

    if response.status_code != 200:
        raise Exception(f"모델 변경 실패: {response.status_code} - {response.text}")
    return True


def check_ollama_connection():
    """Ollama 서버 연결 확인"""
    try:
        # 간단한 테스트 요청
        response = requests.post(OLLAMA_API_URL,
                                 json={"model": DEFAULT_MODEL, "prompt": "test", "stream": False},
                                 timeout=5)
        return response.status_code == 200
    except:
        return False


def check_sd_connection():
    """Stable Diffusion 서버 연결 확인"""
    try:
        response = requests.get(f"{SD_API_URL}/sd-models", timeout=5)
        return response.status_code == 200
    except:
        return False


def check_tts_connection():
    """TTS 서버 연결 확인"""
    try:
        # 간단한 테스트 요청 (빈 텍스트로는 오류가 날 수 있으므로 최소한의 텍스트 사용)
        test_payload = {
            "text": "테스트",
            "text_lang": "ko",
            "ref_audio_path": "A-A3-E-055-0101.wav",
            "prompt_lang": "ko",
            "prompt_text": "테스트",
            "media_type": "wav"
        }
        response = requests.post(TTS_API_URL, json=test_payload, timeout=10)
        return response.status_code == 200
    except:
        return False


if __name__ == '__main__':
    print("=== 통합 AI 서버 시작 ===")
    print(f"Ollama API URL: {OLLAMA_API_URL}")
    print(f"Stable Diffusion API URL: {SD_API_URL}")
    print(f"TTS API URL: {TTS_API_URL}")
    print("서버 포트: 8000")

    # 연결 테스트
    print("\n=== 연결 테스트 ===")

    # Ollama 연결 테스트
    if check_ollama_connection():
        print("✅ Ollama API 연결 성공!")
    else:
        print("⚠️ Ollama API 연결 실패 - 서버가 실행 중인지 확인하세요")

    # Stable Diffusion 연결 테스트
    if check_sd_connection():
        try:
            models = get_sd_models()
            model_names = [model["title"] for model in models]
            print(f"✅ Stable Diffusion API 연결 성공! 사용 가능한 모델: {model_names[:3]}{'...' if len(model_names) > 3 else ''}")
        except Exception as e:
            print(f"⚠️ Stable Diffusion 모델 목록 조회 실패: {e}")
    else:
        print("⚠️ Stable Diffusion API 연결 실패 - AUTOMATIC1111 웹 UI가 --api 옵션으로 실행 중인지 확인하세요")

    # TTS 연결 테스트
    if check_tts_connection():
        print("✅ TTS API 연결 성공!")
    else:
        print("⚠️ TTS API 연결 실패 - TTS 서버가 localhost:9880에서 실행 중인지 확인하세요")

    print("\n=== 서버 실행 ===")
    app.run(host='0.0.0.0', port=8000, debug=True)
