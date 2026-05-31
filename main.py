from fastapi import FastAPI, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from groq import Groq
import tempfile, base64, os, uuid
from dotenv import load_dotenv
from gtts import gTTS
import uvicorn
from pathlib import Path

# Load environment variables
load_dotenv()

# Initialize FastAPI
app = FastAPI(title="AI Voice Assistant API")

# Get API key from environment
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    print("❌ ERROR: GROQ_API_KEY not found in .env file")
    print("Please create a .env file with: GROQ_API_KEY=your_key_here")
    print("Get your API key from: https://console.groq.com")
    api_key = None
else:
    print("✅ GROQ_API_KEY loaded successfully")
    print(f"🔑 API Key starts with: {api_key[:8]}...")

# Groq Client - Simplified initialization without proxies parameter
try:
    if api_key:
        # Simple initialization - no extra parameters
        client = Groq(api_key=api_key)
        print("✅ Groq client initialized successfully")
    else:
        client = None
except Exception as e:
    print(f"❌ Error initializing Groq client: {e}")
    print("⚠️  Make sure you have the latest groq package: pip install --upgrade groq")
    client = None

# CORS middleware - Allow all origins for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# SYSTEM PROMPT
SYSTEM_PROMPT = """
You are a helpful AI Voice Assistant. Keep your answers:
- Short and concise (max 2-3 sentences)
- Factual and accurate
- Easy to understand

If you don't know something, say so.
If the user's question is unclear, ask for clarification.

Examples:
User: What is the weather like?
Assistant: I don't have access to real-time weather data. Please check a weather website or app.

User: Tell me about yourself
Assistant: I'm an AI voice assistant powered by Groq. I can answer questions and have simple conversations.
"""

# Store conversation history
conversation_history = {}

# MAIN ENDPOINT
@app.post("/assistant")
async def assistant_api(file: UploadFile, session_id: str = Form("default")):
    try:
        print(f"\n📝 New request from session: {session_id}")
        print(f"📁 File received: {file.filename}, Type: {file.content_type}")
        
        # Check if client is initialized
        if not client:
            error_msg = "API client not initialized. Check GROQ_API_KEY in .env file"
            print(f"❌ {error_msg}")
            return JSONResponse(
                status_code=500,
                content={"error": error_msg}
            )
        
        # Step 1: Transcribe audio
        user_text = await transcribe(file)
        print(f"🗣️ User said: {user_text}")
        
        if not user_text or user_text.strip() == "":
            error_response = "I couldn't hear anything. Please speak clearly and try again."
            return JSONResponse({
                "transcript": "",
                "response": error_response,
                "audio_base64": await tts(error_response)
            })
        
        # Step 2: Get AI response
        ai_reply = await get_ai_reply(user_text, session_id)
        print(f"🤖 AI response: {ai_reply}")
        
        # Step 3: Convert to speech
        ai_voice_base64 = await tts(ai_reply)
        print("✅ Audio generated successfully")
        
        # Store in history
        if session_id not in conversation_history:
            conversation_history[session_id] = []
        conversation_history[session_id].append({
            "user": user_text,
            "assistant": ai_reply
        })
        
        return JSONResponse({
            "transcript": user_text,
            "response": ai_reply,
            "audio_base64": ai_voice_base64
        })
        
    except Exception as e:
        print(f"❌ Error in assistant_api: {str(e)}")
        import traceback
        traceback.print_exc()
        
        # Return a friendly error message
        error_message = "Sorry, I encountered an error. Please try again."
        try:
            error_audio = await tts(error_message)
        except:
            error_audio = ""
            
        return JSONResponse(
            status_code=200,
            content={
                "transcript": "Error processing audio",
                "response": error_message,
                "audio_base64": error_audio
            }
        )

# TRANSCRIBE AUDIO FILE
async def transcribe(file: UploadFile):
    temp_path = None
    try:
        content = await file.read()
        
        if len(content) == 0:
            print("❌ Empty audio file received")
            return ""
        
        print(f"📊 Audio size: {len(content)} bytes")
        
        # Save to temp file with correct extension
        suffix = ".webm"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_audio:
            temp_audio.write(content)
            temp_path = temp_audio.name
            print(f"💾 Saved temp file: {temp_path}")

        print("🎤 Transcribing audio with Whisper...")
        
        # Open and transcribe
        with open(temp_path, "rb") as audio_file:
            transcript = client.audio.transcriptions.create(
                model="whisper-large-v3",
                file=audio_file,
                response_format="text"
            )
        
        # Handle different response formats
        if hasattr(transcript, 'text'):
            result = transcript.text.strip()
        else:
            result = str(transcript).strip()
        
        print(f"📝 Transcription result: '{result}'")
        return result
        
    except Exception as e:
        print(f"❌ Error in transcribe: {str(e)}")
        import traceback
        traceback.print_exc()
        return ""
    finally:
        # Clean up temp file
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
                print(f"🧹 Cleaned up temp file: {temp_path}")
            except:
                pass

# GET CHAT RESPONSE
async def get_ai_reply(user_prompt: str, session_id: str = "default"):
    try:
        print("🤔 Fetching response from Groq AI...")
        
        # Get conversation history (last 3 messages for context)
        history = conversation_history.get(session_id, [])[-3:]
        
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ]
        
        # Add conversation history
        for turn in history:
            messages.append({"role": "user", "content": turn["user"]})
            messages.append({"role": "assistant", "content": turn["assistant"]})
        
        # Add current user prompt
        messages.append({"role": "user", "content": user_prompt})
        
        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            temperature=0.3,
            max_tokens=100,
            messages=messages,
        )
        
        response = completion.choices[0].message.content.strip()
        print(f"💬 Got response: {response[:50]}...")
        return response
        
    except Exception as e:
        print(f"❌ Error in get_ai_reply: {str(e)}")
        return "I'm having trouble processing your request right now. Please try again."

# TEXT TO SPEECH (gTTS)
async def tts(text: str):
    temp_path = None
    try:
        print(f"🔊 Converting to speech: {text[:50]}...")
        
        if not text or text.strip() == "":
            text = "I didn't catch that. Could you please repeat?"
        
        # Create temp file with unique name
        temp_path = f"tts_temp_{uuid.uuid4().hex}.mp3"
        
        # Generate speech
        tts_obj = gTTS(
            text=text, 
            lang="en", 
            tld="com",
            slow=False
        )
        
        tts_obj.save(temp_path)
        print(f"💾 Saved TTS to: {temp_path}")
        
        # Read and encode
        with open(temp_path, "rb") as f:
            audio_bytes = f.read()
        
        audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
        print(f"✅ Audio encoded, size: {len(audio_b64)} chars")
        
        return audio_b64
        
    except Exception as e:
        print(f"❌ Error in tts: {str(e)}")
        return ""
    finally:
        # Clean up temp file
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
                print(f"🧹 Cleaned up TTS file: {temp_path}")
            except:
                pass

# Health check endpoint
@app.get("/health")
async def health_check():
    return {
        "status": "healthy", 
        "api_key_configured": client is not None,
        "groq_client": "initialized" if client else "not initialized"
    }

# Test endpoint
@app.get("/test")
async def test():
    return {"message": "Voice Assistant API is running!"}

# Root endpoint
@app.get("/")
async def root():
    return {
        "name": "AI Voice Assistant API",
        "version": "1.0",
        "endpoints": {
            "POST /assistant": "Process voice input",
            "GET /health": "Health check",
            "GET /test": "Test endpoint",
            "GET /docs": "API documentation"
        }
    }

# Function to check environment
def check_environment():
    """Check if environment is properly set up"""
    print("\n🔍 Checking environment...")
    
    # Check if .env exists
    if not Path(".env").exists():
        print("❌ .env file not found")
        print("   Creating .env file...")
        with open(".env", "w") as f:
            f.write("# Groq API Key - Get yours from https://console.groq.com\n")
            f.write("GROQ_API_KEY=your_api_key_here\n")
        print("   ✅ .env file created")
        print("   ⚠️  Please edit .env and add your Groq API key")
        return False
    
    # Check if API key is set
    with open(".env", "r") as f:
        content = f.read()
        if "your_api_key_here" in content or "GROQ_API_KEY=" not in content:
            print("❌ GROQ_API_KEY not set in .env file")
            print("   Please edit .env and add your Groq API key")
            return False
    
    return True

# Main execution
if __name__ == "__main__":
    print("\n" + "="*60)
    print("🎙️  AI Voice Assistant Server")
    print("="*60)
    
    # Check environment
    if not check_environment():
        print("\n⚠️  Please fix the issues above and try again")
        exit(1)
    
    print("\n✅ Configuration loaded successfully")
    print(f"📡 Server will start on: http://localhost:8000")
    print("\n📝 Available endpoints:")
    print("   - POST http://localhost:8000/assistant")
    print("   - GET  http://localhost:8000/health")
    print("   - GET  http://localhost:8000/test")
    print("   - GET  http://localhost:8000/docs")
    print("\n🛑 Press Ctrl+C to stop the server")
    print("="*60 + "\n")
    
    # Run the server
    try:
        uvicorn.run(
            "main:app",
            host="0.0.0.0",
            port=8000,
            reload=True,
            log_level="info"
        )
    except KeyboardInterrupt:
        print("\n\n👋 Server stopped by user")
    except Exception as e:
        print(f"\n❌ Error starting server: {e}")