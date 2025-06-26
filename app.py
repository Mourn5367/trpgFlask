from flask import Flask
from flask_sock import Sock
import json
import requests
import traceback
import uuid
import base64
from datetime import datetime
from PIL import Image
import io

app = Flask(__name__)
sock = Sock(app)

# API 엔드포인트 설정
OLLAMA_API_URL = "http://192.168.24.189:11434/api/chat"  # /api/chat 사용
SD_API_URL = "http://192.168.24.189:7860/sdapi/v1"
TTS_API_URL = "http://192.168.24.189:9880/tts"

# 기본 설정
DEFAULT_MODEL = "gemma3:4b"
# 메모리 기반 대화 세션 저장소
chat_sessions = {}
# 시스템 프롬프트 설정
SYSTEM_PROMPTS = {
    "default": """# TRPG MASTER의 역할

    당신은 TRPG 스토리를 생성하는 작가이자, TRPG 게임을 진행하는 GAME MASTER 입니다.
    
    ## 글쓰기 스타일
    - **감정 표현**: 풍부한 감정과 분위기를 담아주세요
    - **생동감**: 독자가 몰입할 수 있는 생생한 묘사를 하세요
    - **창의성**: 독창적이고 참신한 아이디어를 제시하세요
    
    ## 답변 방식
    - 플레이어에게는 체력에 해당하는 HP와 랜덤한 능력치, 특별한 기질을 가지고 있습니다
    - 플레이어의 상황과 능력에 맞게 스토리를 진행합니다
    - 세계관은 현실, 가상 모두 가능합니다
    - 플레이어의 응답에 따라 이야기를 생성하시오
    - 이해가 가지 않는 부분과 없는 지식일 경우 플레이어에게 확답을 받으시오
    - 마지막에는 플레이어의 선택지를 약 3개정도 주도록 하시오. 선택지가 적거나 없는 경우 억지로 3개를 맞출필요는 없습니다.
    - 확률이 들어가는 선택지인 경우 플레이어에게 1 부터 10 까지 나오는 주사위를 던지게 하시오
    - 특별한 기질을 가지고 있는 경우 플레이어에게 유리한 주사위 숫자가 나오도록합니다.
    
    ## 답변 구조
    0. 플레이어에게 선택지를 제공하여 적합한 응답을 한 경우 응답에 맞는 이벤트 발생
    1. 플레이어가 잘못된 응답을 한 경우 답변을 중지하고 다시 받도록합니다.
    2. 진행중인 이벤트 설명
    3. 플레이어에게 닥친 상황 설명
    4. 플레이어에게 선택지 제공
    
    """,
}


def get_system_prompt(original_request):
    """요청에 따른 시스템 프롬프트 반환"""
    # 요청에서 프롬프트 타입 확인
    prompt_type = original_request.get("prompt_type", "default")

    # 커스텀 시스템 프롬프트가 있으면 우선 사용
    custom_prompt = original_request.get("system_prompt")
    if custom_prompt:
        return custom_prompt

    # 사전 정의된 프롬프트 사용
    return SYSTEM_PROMPTS.get(prompt_type, SYSTEM_PROMPTS["default"])


class ChatSession:
    def __init__(self, conversation_id=None):
        # Spring Boot에서 전달받은 세션 ID 사용
        self.conversation_id = conversation_id or str(uuid.uuid4())
        self.messages = []
        self.created_at = datetime.now()

    def add_message(self, role, content):
        """메시지 추가 (role: 'user' 또는 'assistant')"""
        message = {
            "role": role,
            "content": content,
            "order": len(self.messages),
            "timestamp": datetime.now().isoformat()
        }
        self.messages.append(message)
        return message

    def get_messages_for_ollama(self):
        """Ollama API용 메시지 형식으로 변환"""
        return [{"role": msg["role"], "content": msg["content"]} for msg in self.messages]

    def get_conversation_info(self):
        """대화 정보 반환"""
        return {
            "conversation_id": self.conversation_id,
            "message_count": len(self.messages),
            "created_at": self.created_at.isoformat(),
            "messages": self.messages
        }


def get_or_create_session(conversation_id=None):
    """세션 가져오기 또는 새로 생성 (Spring Boot 세션 ID 기반)"""
    if conversation_id and conversation_id in chat_sessions:
        print(f"기존 세션 사용: {conversation_id}")
        return chat_sessions[conversation_id]

    # 새 세션 생성 (Spring Boot에서 전달받은 ID 사용)
    session = ChatSession(conversation_id)
    chat_sessions[session.conversation_id] = session
    print(f"새 세션 생성: {session.conversation_id}")
    return session


@app.route('/')
def index():
    return f"""
    <h1>통합 AI 서버 (대화 기록 관리)</h1>
    <p>WebSocket 서버가 실행 중입니다.</p>
    <ul>
        <li>채팅: Ollama /api/chat 엔드포인트 (대화 기록 관리)</li>
        <li>이미지 생성: Stable Diffusion API 연동</li>
        <li>음성 생성: TTS API 연동</li>
        <li>WebSocket 엔드포인트: /ws</li>
    </ul>
    <h3>현재 활성 대화 수: {len(chat_sessions)}</h3>
    <h3>서비스 상태:</h3>
    <ul>
        <li>Ollama: {'✅' if check_ollama_connection() else '❌'}</li>
        <li>Stable Diffusion: {'✅' if check_sd_connection() else '❌'}</li>
        <li>TTS: {'✅' if check_tts_connection() else '❌'}</li>
    </ul>
    """


@app.route('/conversations')
def list_conversations():
    """활성 대화 목록 반환 (디버깅용)"""
    conversations = []
    for session_id, session in chat_sessions.items():
        conversations.append({
            "conversation_id": session_id,
            "message_count": len(session.messages),
            "created_at": session.created_at.isoformat(),
            "last_message": session.messages[-1]["content"][:50] + "..." if session.messages else "대화 없음"
        })
    return json.dumps(conversations, ensure_ascii=False, indent=2)


@app.route('/conversation/<conversation_id>')
def get_conversation(conversation_id):
    """특정 대화 내용 반환 (디버깅용)"""
    if conversation_id in chat_sessions:
        return json.dumps(chat_sessions[conversation_id].get_conversation_info(), ensure_ascii=False, indent=2)
    else:
        return json.dumps({"error": "대화를 찾을 수 없습니다."}, ensure_ascii=False), 404


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

                # 채팅 관련 처리 (Ollama) - 대화 기록 관리 포함
                if msg_type == "chat":
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
                        "active_conversations": len(chat_sessions),
                        "services": {
                            "ollama": check_ollama_connection(),
                            "stable_diffusion": check_sd_connection(),
                            "tts": check_tts_connection()
                        },
                        "original": json_data
                    }))

                # 이전 버전 호환성 (prompt만 있는 경우)
                elif "prompt" in json_data and not msg_type:
                    json_data["type"] = "generate_image"
                    handle_image_generation(ws, json_data)

                else:
                    ws.send(json.dumps({
                        "status": "error",
                        "message": f"지원하지 않는 메시지 타입: {msg_type}",
                        "supported_types": [
                            "chat",  # Ollama 채팅 전용
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
    """채팅 메시지 처리 (대화 기록 관리 포함)"""
    message = json_data.get("message", "")
    model = json_data.get("model", DEFAULT_MODEL)
    conversation_id = json_data.get("conversation_id")

    print(f"채팅 처리: '{message}', 모델: {model}")

    if not message.strip():
        ws.send(json.dumps({
            "status": "error",
            "error": "메시지가 비어있습니다.",
            "original": json_data
        }))
        return

    # 세션 가져오기 또는 생성
    session = get_or_create_session(conversation_id)

    # 사용자 메시지 추가
    user_msg = session.add_message("user", message)

    print(f"대화 ID: {session.conversation_id}")
    print(f"메시지 순서: {user_msg['order']}")
    print(f"현재 메시지 수: {len(session.messages)}")

    # Ollama 스트리밍 호출 (대화 기록 포함)
    print("Ollama API 스트리밍 호출 시작...")
    stream_ollama_chat(session, model, ws, json_data)
    print("Ollama API 스트리밍 호출 완료")


def handle_image_generation(ws, json_data):
    """이미지 생성 처리 (Stable Diffusion API)"""
    prompt = json_data.get("prompt", "")

    if not prompt:
        ws.send(json.dumps({
            "status": "error",
            "message": "프롬프트가 필요합니다.",
            "original": json_data
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


def handle_list_conversations(ws, json_data):
    """대화 목록 반환 (사용 안 함 - Spring Boot에서 관리)"""
    ws.send(json.dumps({
        "status": "error",
        "message": "이 기능은 Spring Boot에서 관리합니다.",
        "note": "Flask는 순수 AI 처리만 담당합니다."
    }))


def handle_load_conversation(ws, json_data):
    """특정 대화 불러오기 (사용 안 함 - Spring Boot에서 관리)"""
    ws.send(json.dumps({
        "status": "error",
        "message": "이 기능은 Spring Boot에서 관리합니다.",
        "note": "세션 ID로 바로 채팅하세요."
    }))


def handle_new_conversation(ws, json_data):
    """새 대화 시작 (사용 안 함 - Spring Boot에서 관리)"""
    ws.send(json.dumps({
        "status": "error",
        "message": "이 기능은 Spring Boot에서 관리합니다.",
        "note": "새 세션 ID로 바로 채팅하세요."
    }))


def stream_ollama_chat(session, model, ws, original_request):
    """Ollama /api/chat 엔드포인트를 사용하여 스트리밍 (대화 기록 포함)"""

    # 시스템 프롬프트 설정
    system_prompt = get_system_prompt(original_request)

    # messages 배열 구성 (시스템 프롬프트 포함)
    messages = []

    # 시스템 프롬프트가 있으면 첫 번째에 추가
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    # 기존 대화 기록 추가
    messages.extend(session.get_messages_for_ollama())

    payload = {
        "model": model,
        "messages": messages,
        "stream": True
    }

    print(f"Ollama /api/chat 요청: {payload}")

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

                    if 'message' in chunk_data and 'content' in chunk_data['message']:
                        chunk_text = chunk_data['message']['content']
                        full_response += chunk_text

                        ws.send(json.dumps({
                            "status": "streaming",
                            "type": "chat_chunk",
                            "conversation_id": session.conversation_id,
                            "chunk": chunk_text,
                            "response_so_far": full_response,
                            "done": False,
                            "chunk_number": chunk_count,
                            "message_order": len(session.messages),
                            "original": original_request
                        }))

                    if chunk_data.get('done', False):
                        print("스트리밍 완료됨")

                        # 어시스턴트 응답을 세션에 추가
                        assistant_msg = session.add_message("assistant", full_response)

                        ws.send(json.dumps({
                            "status": "complete",
                            "type": "chat_complete",
                            "conversation_id": session.conversation_id,
                            "response": full_response,
                            "done": True,
                            "model": model,
                            "assistant_message_order": assistant_msg['order'],
                            "total_messages": len(session.messages),
                            "total_duration": chunk_data.get("total_duration", 0),
                            "load_duration": chunk_data.get("load_duration", 0),
                            "prompt_eval_count": chunk_data.get("prompt_eval_count", 0),
                            "eval_count": chunk_data.get("eval_count", 0),
                            "total_chunks": chunk_count,
                            "original": original_request
                        }))
                        break

                except json.JSONDecodeError:
                    print(f"JSON 파싱 오류 (청크): {line}")
                    continue

        print(f"총 {chunk_count}개 청크 처리 완료")
        print(f"세션 {session.conversation_id}에 총 {len(session.messages)}개 메시지 저장됨")

    except requests.exceptions.RequestException as e:
        error_msg = f"Ollama API 호출 오류: {str(e)}"
        print(error_msg)
        ws.send(json.dumps({
            "status": "error",
            "type": "chat_error",
            "conversation_id": session.conversation_id,
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
        response = requests.post(OLLAMA_API_URL,
                                 json={"model": DEFAULT_MODEL, "messages": [{"role": "user", "content": "test"}],
                                       "stream": False},
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
    print("=== 통합 AI 서버 (대화 기록 관리) 시작 ===")
    print(f"Ollama API URL: {OLLAMA_API_URL}")
    print(f"Stable Diffusion API URL: {SD_API_URL}")
    print(f"TTS API URL: {TTS_API_URL}")
    print("서버 포트: 8000")
    print("대화 기록: 메모리 기반 저장")

    # 연결 테스트
    print("\n=== 연결 테스트 ===")
    print(f"✅ Ollama API: {'연결 성공' if check_ollama_connection() else '❌ 연결 실패'}")
    print(f"✅ Stable Diffusion API: {'연결 성공' if check_sd_connection() else '❌ 연결 실패'}")
    print(f"✅ TTS API: {'연결 성공' if check_tts_connection() else '❌ 연결 실패'}")

    app.run(host='0.0.0.0', port=8000, debug=True)
