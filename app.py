from flask import Flask, request, jsonify, Response, render_template, session
from flask_cors import CORS
from langchain_ollama import OllamaLLM
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
import json
import requests
import uuid
from datetime import datetime, timedelta

app = Flask(__name__)
app.secret_key = 'your-secret-key-change-this-in-production'
CORS(app)

# Ollama LLM 초기화
llm = OllamaLLM(
    model="gemma3:4b",
    base_url="http://localhost:11434",
    temperature=0.7
)

# 사용자별 메모리 저장소
user_memories = {}
user_conversations = {}

def get_user_id():
    """사용자 ID 가져오기 또는 생성"""
    if 'user_id' not in session:
        session['user_id'] = str(uuid.uuid4())
        print(f"새로운 사용자 세션 생성: {session['user_id'][:8]}...")
    return session['user_id']

def get_user_memory(user_id):
    """사용자별 메모리 가져오기 또는 생성"""
    try:
        if user_id not in user_memories:
            print(f"사용자 {user_id[:8]}...의 새로운 메모리 생성")
            user_memories[user_id] = ConversationBufferMemory(return_messages=True)
            user_conversations[user_id] = ConversationChain(
                llm=llm,
                memory=user_memories[user_id],
                verbose=True
            )
        
        return user_memories[user_id], user_conversations[user_id]
    
    except Exception as e:
        print(f"메모리 생성 중 오류 ({user_id[:8]}...): {str(e)}")
        # 오류 발생 시 강제로 새 메모리 생성
        user_memories[user_id] = ConversationBufferMemory(return_messages=True)
        user_conversations[user_id] = ConversationChain(
            llm=llm,
            memory=user_memories[user_id],
            verbose=True
        )
        return user_memories[user_id], user_conversations[user_id]

@app.route('/')
def index():
    """메인 페이지"""
    user_id = get_user_id()
    return render_template('index.html')

@app.route('/user_info')
def user_info():
    """사용자 정보 조회"""
    user_id = get_user_id()
    memory_count = 0
    
    if user_id in user_memories:
        try:
            memory_count = len(user_memories[user_id].chat_memory.messages)
        except:
            memory_count = 0
    
    return jsonify({
        "user_id": user_id,
        "memory_count": memory_count,
        "total_users": len(user_memories)
    })

@app.route('/test', methods=['GET'])
def test_connection():
    """연결 테스트"""
    user_id = get_user_id()
    try:
        response = llm.invoke("안녕하세요!")
        return jsonify({
            "status": "success",
            "response": response,
            "message": f"Ollama 연결 성공! (사용자: {user_id[:8]}...)",
            "user_id": user_id
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"연결 실패: {str(e)}",
            "user_id": user_id
        }), 500

@app.route('/prompt_engineering', methods=['POST'])
def prompt_engineering():
    """프롬프트 엔지니어링 - 사용자별 메모리 지원"""
    user_id = get_user_id()
    
    try:
        data = request.json
        prompt_template = data.get('prompt', '')
        user_text = data.get('text', '')
        use_memory = data.get('use_memory', False)
        
        print(f"프롬프트 실행: 사용자 {user_id[:8]}..., 메모리 사용: {use_memory}")
        
        if use_memory:
            # 사용자별 메모리 사용
            try:
                user_memory, user_conversation = get_user_memory(user_id)
                
                if "{text}" in prompt_template:
                    final_prompt = prompt_template.replace("{text}", user_text)
                else:
                    final_prompt = f"{prompt_template}\n\n{user_text}"
                
                def generate_with_memory():
                    try:
                        print(f"메모리 모드 실행 중: {user_id[:8]}...")
                        response = user_conversation.predict(input=final_prompt)
                        
                        # 단어 단위로 스트리밍
                        words = response.split()
                        for word in words:
                            yield f"data: {json.dumps({'token': word + ' '})}\n\n"
                        
                        yield f"data: {json.dumps({'done': True})}\n\n"
                        print(f"메모리 모드 완료: {user_id[:8]}...")
                        
                    except Exception as e:
                        error_msg = f"메모리 처리 오류: {str(e)}"
                        print(error_msg)
                        yield f"data: {json.dumps({'error': error_msg})}\n\n"
                
                return Response(generate_with_memory(), 
                              mimetype='text/event-stream',
                              headers={'Cache-Control': 'no-cache'})
            
            except Exception as e:
                error_msg = f"메모리 초기화 오류: {str(e)}"
                print(error_msg)
                return jsonify({"error": error_msg, "status": "error"}), 500
        
        else:
            # 메모리 없는 직접 스트리밍
            final_prompt = prompt_template.replace("{text}", user_text)
            
            def generate_direct():
                try:
                    print(f"직접 모드 실행 중: {user_id[:8]}...")
                    response = requests.post(
                        "http://localhost:11434/api/generate",
                        json={
                            "model": "gemma3:4b",
                            "prompt": final_prompt,
                            "stream": True
                        },
                        stream=True
                    )
                    
                    for line in response.iter_lines():
                        if line:
                            try:
                                chunk = json.loads(line.decode('utf-8'))
                                if 'response' in chunk:
                                    yield f"data: {json.dumps({'token': chunk['response']})}\n\n"
                                if chunk.get('done', False):
                                    yield f"data: {json.dumps({'done': True})}\n\n"
                                    break
                            except json.JSONDecodeError:
                                continue
                    
                    print(f"직접 모드 완료: {user_id[:8]}...")
                    
                except Exception as e:
                    error_msg = f"직접 처리 오류: {str(e)}"
                    print(error_msg)
                    yield f"data: {json.dumps({'error': error_msg})}\n\n"
            
            return Response(generate_direct(), 
                          mimetype='text/event-stream',
                          headers={'Cache-Control': 'no-cache'})
        
    except Exception as e:
        error_msg = f"전체 처리 오류: {str(e)}"
        print(error_msg)
        return jsonify({
            "error": error_msg,
            "status": "error",
            "user_id": user_id
        }), 500

@app.route('/clear_prompt_memory', methods=['POST'])
def clear_prompt_memory():
    """사용자별 메모리 초기화"""
    user_id = get_user_id()
    
    try:
        # 기존 메모리 완전 삭제
        if user_id in user_memories:
            del user_memories[user_id]
            print(f"기존 메모리 삭제: {user_id[:8]}...")
        
        if user_id in user_conversations:
            del user_conversations[user_id]
            print(f"기존 대화 체인 삭제: {user_id[:8]}...")
        
        # 새로운 메모리 생성
        user_memories[user_id] = ConversationBufferMemory(return_messages=True)
        user_conversations[user_id] = ConversationChain(
            llm=llm,
            memory=user_memories[user_id],
            verbose=True
        )
        
        print(f"새로운 메모리 생성 완료: {user_id[:8]}...")
        
        return jsonify({
            "status": "success",
            "message": f"사용자 {user_id[:8]}...의 메모리가 초기화되었습니다.",
            "user_id": user_id
        })
        
    except Exception as e:
        error_msg = f"메모리 초기화 오류: {str(e)}"
        print(error_msg)
        return jsonify({
            "error": error_msg,
            "status": "error",
            "user_id": user_id
        }), 500

if __name__ == '__main__':
    print("🚀 프롬프트 엔지니어링 Flask 서버 시작")
    print("📡 브라우저에서 http://localhost:5000 으로 접근하세요")
    print("👥 사용자별 독립적인 메모리 지원")
    print("🔍 디버깅 로그 활성화")
    app.run(debug=True, host='0.0.0.0', port=5000)
