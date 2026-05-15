# app.py -- Full backend with AES-GCM encryption, Ollama + MiniLM classification, and Email OTP for Register/Login
import os
import json
import hashlib
import traceback
import base64
import smtplib
import random
from datetime import datetime, timedelta
from typing import Tuple, Optional
from email.message import EmailMessage

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware

from fastapi import FastAPI, Form, Request
from fastapi.responses import JSONResponse, FileResponse
# (your other imports here)

app = FastAPI()

# ✅ ADD THIS BLOCK — NOTHING ELSE
from fastapi.staticfiles import StaticFiles
BASE = os.path.dirname(os.path.abspath(__file__))
app.mount("/", StaticFiles(directory=BASE, html=True), name="static")


from io import BytesIO
import PyPDF2
from pdf2image import convert_from_bytes
import pytesseract
from PIL import Image
import requests
import re

# Crypto
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

# sentence-transformers fallback
from sentence_transformers import SentenceTransformer, util

# reportlab (export)
try:
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas
    REPORTLAB_AVAILABLE = True
except Exception:
    REPORTLAB_AVAILABLE = False

# ---------------- Config ----------------
BASE = os.path.dirname(os.path.abspath(__file__))
USER_DATA = os.path.join(BASE, "user_data")
USERS_FILE = os.path.join(BASE, "users.json")
CHAIN_FILE = os.path.join(BASE, "data", "global_chain.json")
LOGIN_LOG = os.path.join(BASE, "data", "login_log.json")
DEBUG_FILE = os.path.join(BASE, "data", "debug_last.json")
OTP_CACHE = os.path.join(BASE, "data", "otp_cache.json")
MASTER_KEY_PATH = os.environ.get("MASTER_KEY_PATH", os.path.join(BASE, "data", "master_key.bin"))

# SMTP config (set as env vars or edit here)
SMTP_SERVER = os.environ.get("SMTP_SERVER", "smtp.gmail.com")
SMTP_PORT = int(os.environ.get("SMTP_PORT", "587"))
SMTP_USER = os.environ.get("SMTP_USER", "barkathshaik2004@gmail.com")
SMTP_PASS = os.environ.get("SMTP_PASS", "slgjnczytmunqspc")  # use app password with Gmail

os.makedirs(USER_DATA, exist_ok=True)
os.makedirs(os.path.join(BASE, "data"), exist_ok=True)

# ---------------- Utils ----------------
def safe_load_json(path, default):
    try:
        if not os.path.exists(path):
            return default
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[WARN] load json fail {path}: {e}")
        return default

def safe_save_json(path, data):
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)
    except Exception as e:
        print(f"[ERROR] save json fail {path}: {e}")

def sha_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

def sha_str(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()

# ---------------- Normalize users ----------------
def normalize_users_file():
    raw = safe_load_json(USERS_FILE, {})
    if not isinstance(raw, dict):
        raw = {}
    normalized = {}
    changed = False
    for k, v in raw.items():
        if isinstance(v, str):
            normalized[k] = {"password_hash": sha_str(v), "role": "user", "email": None}
            changed = True
        elif isinstance(v, dict):
            ph = v.get("password_hash") or (sha_str(v.get("password")) if v.get("password") else None)
            role = v.get("role", "user")
            email = v.get("email") if v.get("email") else v.get("email_address") if v.get("email_address") else None
            if ph is None:
                continue
            normalized[k] = {"password_hash": ph, "role": role, "email": email}
            if "password" in v or "role" not in v:
                changed = True
        else:
            changed = True
    if normalized and not any(u.get("role") == "admin" for u in normalized.values()):
        first_user = list(normalized.keys())[0]
        normalized[first_user]["role"] = "admin"
        changed = True
    if changed:
        safe_save_json(USERS_FILE, normalized)
    return normalized

_users_cache = normalize_users_file()

# ---------------- MiniLM fallback ----------------
try:
    MINI_MODEL = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    LABELS = [
        "resume","certificate","invoice","bill","novel","official letter",
        "id document","marksheet","degree","passport","contract","report","article","unknown"
    ]
    LABEL_EMBS = MINI_MODEL.encode(LABELS, convert_to_tensor=True)
    def classify_minilm(text: str):
        emb = MINI_MODEL.encode(text, convert_to_tensor=True)
        sims = util.cos_sim(emb, LABEL_EMBS)[0]
        best = sims.argmax().item()
        return LABELS[best], float(sims[best].item())
    print("[INFO] MiniLM loaded.")
except Exception as e:
    print("[WARN] MiniLM load failed:", e)
    MINI_MODEL = None
    def classify_minilm(text: str):
        return "unknown", 0.0

# ---------------- Ollama config ----------------
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
LLAMA_MODEL = os.environ.get("LLAMA_MODEL", "llama3.1:8b")

def save_debug(obj: dict):
    try:
        safe_save_json(DEBUG_FILE, obj)
    except Exception as e:
        print("[WARN] failed to save debug:", e)

# ---------------- OCR helpers ----------------
def extract_text_from_pdf_bytes(pdf_bytes: bytes, max_pages: int = 10):
    text = ""
    try:
        reader = PyPDF2.PdfReader(BytesIO(pdf_bytes))
        for page in reader.pages:
            try:
                p = page.extract_text()
                if p:
                    text += p + "\n"
            except Exception:
                continue
    except Exception as e:
        print("[DEBUG] PyPDF2 read failed:", e)

    if len(text.strip()) < 80:
        try:
            images = convert_from_bytes(pdf_bytes, first_page=1, last_page=min(max_pages, 10))
            ocr_text = ""
            for img in images:
                if img.mode != "RGB":
                    img = img.convert("RGB")
                ocr_text += pytesseract.image_to_string(img) + "\n"
            if ocr_text.strip():
                text = (text + "\n" + ocr_text).strip()
        except Exception as e:
            print("[WARN] OCR fallback failed:", e)
            save_debug({"stage":"ocr_error","error": str(e)})
    return text.strip()

# ---------------- Ollama call (robust JSON-safe) ----------------
async def classify_document_llm_ollama(text: str) -> Tuple[Optional[str], Optional[float], Optional[str]]:
    """
    Offline document classification using Ollama (local llama model)
    Returns (label, confidence, explanation)
    """
    print(f"\n{'='*60}")
    print(f"[OLLAMA DEBUG] Starting classification...")
    print(f"[OLLAMA DEBUG] Text length: {len(text)} chars")
    print(f"[OLLAMA DEBUG] Text preview: {text[:200]}...")
    print(f"[OLLAMA DEBUG] OLLAMA_URL: {OLLAMA_URL}")
    print(f"[OLLAMA DEBUG] MODEL: {LLAMA_MODEL}")
    print(f"{'='*60}\n")
    
    if not text or len(text.strip()) < 20:
        print("[OLLAMA DEBUG] ❌ Text too short, skipping Ollama")
        return None, None, None
    
    try:
        # Truncate to 800 chars for faster processing
        text_sample = text[:800].strip()
        
        prompt = f"""Classify this document in one word.

Text: {text_sample}

Types: resume, certificate, invoice, bill, id, passport, contract, report, letter

JSON: {{"label":"certificate","confidence":0.9,"explanation":"reason"}}"""

        print(f"[OLLAMA DEBUG] Sending request to {OLLAMA_URL}/api/generate...")
        print(f"[OLLAMA DEBUG] Prompt length: {len(prompt)} chars")
        
        import time
        start_time = time.time()
        
        response = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={
                "model": LLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "format": "json",
                "options": {
                    "temperature": 0.1,
                    "num_predict": 50  # Limit output tokens for speed
                }
            },
            timeout=120  # 🔥 INCREASED TO 2 MINUTES
        )
        
        elapsed = time.time() - start_time
        print(f"[OLLAMA DEBUG] Response received in {elapsed:.2f} seconds")
        print(f"[OLLAMA DEBUG] Status code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            response_text = result.get("response", "").strip()
            print(f"[OLLAMA DEBUG] Raw response: {response_text[:300]}")
            
            # Try to parse JSON
            try:
                data = json.loads(response_text)
                label = data.get("label", "unknown")
                confidence = float(data.get("confidence", 0.5))
                explanation = data.get("explanation", "")
                
                print(f"[OLLAMA DEBUG] ✅ SUCCESS: {label} ({confidence})")
                return label, confidence, explanation
                
            except json.JSONDecodeError as je:
                print(f"[OLLAMA DEBUG] ⚠️ JSON parse error: {je}")
                print(f"[OLLAMA DEBUG] Trying regex extraction...")
                
                # Fallback: extract from text
                label_match = re.search(r'"label"\s*:\s*"([^"]+)"', response_text)
                conf_match = re.search(r'"confidence"\s*:\s*([\d.]+)', response_text)
                
                if label_match:
                    label = label_match.group(1)
                    confidence = float(conf_match.group(1)) if conf_match else 0.7
                    print(f"[OLLAMA DEBUG] ✅ Extracted: {label} ({confidence})")
                    return label, confidence, "Classified via Ollama"
                else:
                    print(f"[OLLAMA DEBUG] ❌ Could not extract label")
                    return None, None, None
        else:
            print(f"[OLLAMA DEBUG] ❌ Bad status: {response.status_code}")
            print(f"[OLLAMA DEBUG] Response body: {response.text[:500]}")
            return None, None, None

    except requests.exceptions.Timeout:
        print(f"[OLLAMA DEBUG] ❌ TIMEOUT after 120 seconds!")
        print(f"[OLLAMA DEBUG] Your model is TOO SLOW")
        print(f"[OLLAMA DEBUG] Solution: Run 'ollama pull llama3.2:3b'")
        return None, None, None
        
    except requests.exceptions.ConnectionError as ce:
        print(f"[OLLAMA DEBUG] ❌ CONNECTION ERROR: {ce}")
        print(f"[OLLAMA DEBUG] Is Ollama running? Check 'ollama serve'")
        return None, None, None
        
    except Exception as e:
        print(f"[OLLAMA DEBUG] ❌ EXCEPTION: {e}")
        traceback.print_exc()
        return None, None, None   #Users\user\Downloads\Version 6 (3)\Version 6\Backend>python

# ---------------- AES-256-GCM helpers (Option A master key) ----------------
def generate_and_store_master_key(path: str = MASTER_KEY_PATH) -> bytes:
    key = AESGCM.generate_key(bit_length=256)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(key)
    try:
        os.chmod(path, 0o600)
    except Exception:
        pass
    print(f"[INFO] Master key generated at {path}")
    return key

def load_master_key(path: str = MASTER_KEY_PATH) -> bytes:
    if not os.path.exists(path):
        return generate_and_store_master_key(path)
    with open(path, "rb") as f:
        key = f.read()
    if len(key) != 32:
        raise ValueError("Master key must be 32 bytes (AES-256).")
    return key

def encrypt_bytes(plain: bytes, key: bytes = None) -> str:
    if key is None:
        key = load_master_key()
    aesgcm = AESGCM(key)
    nonce = os.urandom(12)
    ct = aesgcm.encrypt(nonce, plain, associated_data=None)
    payload = nonce + ct
    return base64.b64encode(payload).decode("utf-8")

def decrypt_bytes(b64_payload: str, key: bytes = None) -> bytes:
    if key is None:
        key = load_master_key()
    payload = base64.b64decode(b64_payload)
    if len(payload) < 12 + 16:
        raise ValueError("Invalid payload length")
    nonce = payload[:12]
    ct_and_tag = payload[12:]
    aesgcm = AESGCM(key)
    plain = aesgcm.decrypt(nonce, ct_and_tag, associated_data=None)
    return plain

# ---------------- Storage helpers ----------------
def load_chain():
    return safe_load_json(CHAIN_FILE, [])

def save_chain(chain):
    safe_save_json(CHAIN_FILE, chain)

def load_logs():
    return safe_load_json(LOGIN_LOG, [])

def save_logs(logs):
    safe_save_json(LOGIN_LOG, logs)

# ---------------- OTP helpers ----------------
def _load_otp_cache():
    return safe_load_json(OTP_CACHE, {})

def _save_otp_cache(c):
    safe_save_json(OTP_CACHE, c)

def gen_otp_code():
    return "{:06d}".format(random.randint(0, 999999))

def send_email_otp(to_email: str, subject: str, body: str):
    try:
        msg = EmailMessage()
        msg["From"] = SMTP_USER
        msg["To"] = to_email
        msg["Subject"] = subject
        msg.set_content(body)

        # SMTP with TLS
        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT, timeout=20)
        server.starttls()
        server.login(SMTP_USER, SMTP_PASS)
        server.send_message(msg)
        server.quit()
        return True
    except Exception as e:
        print("[WARN] email send failed:", e)
        return False

def create_and_send_otp(username: str, email: str, context: str = "login", expiry_minutes: int = 5):
    """
    context: 'login' or 'register'
    """
    cache = _load_otp_cache()
    otp = gen_otp_code()
    expiry = (datetime.utcnow() + timedelta(minutes=expiry_minutes)).isoformat()
    cache[username] = {
        "otp": otp,
        "expiry": expiry,
        "email": email,
        "context": context
    }
    _save_otp_cache(cache)
    subj = f"Your {context} OTP for DocumentSecurity"
    body = f"Your {context} one-time code is: {otp}\nThis code will expire in {expiry_minutes} minutes."
    ok = send_email_otp(email, subj, body)
    return ok, otp

def verify_otp(username: str, code: str, context: str):
    cache = _load_otp_cache()
    entry = cache.get(username)
    if not entry:
        return False, "No OTP pending"
    if entry.get("context") != context:
        return False, "OTP context mismatch"
    if entry.get("otp") != code:
        return False, "Wrong code"
    expiry = datetime.fromisoformat(entry.get("expiry"))
    if datetime.utcnow() > expiry:
        del cache[username]
        _save_otp_cache(cache)
        return False, "OTP expired"
    # consumed
    del cache[username]
    _save_otp_cache(cache)
    return True, "Verified"

# ---------------- App ----------------
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- Endpoints (auth + register/login OTP + upload + admin + download + debug) ----------------
@app.post("/register")
async def register(username: str = Form(...), password: str = Form(...), email: str = Form(...)):
    users = safe_load_json(USERS_FILE, {})
    if username in users:
        return JSONResponse({"status": "error", "message": "Username already exists"}, status_code=400)

    # Always generate and send OTP *before* storing user
    ok, otp = create_and_send_otp(username, email, context="register")
    if not ok:
        return JSONResponse({"status": "error", "message": "Failed to send OTP"}, status_code=500)

    # Store user temporarily in otp_cache
    cache = _load_otp_cache()
    cache[username] = {
        "otp": otp,
        "expiry": (datetime.utcnow() + timedelta(minutes=5)).isoformat(),
        "email": email,
        "context": "register",
        "pending_user": {
            "password_hash": sha_str(password),
            "role": "admin" if len(users) == 0 else "user",
            "email": email
        }
    }
    _save_otp_cache(cache)

    return {"status": "ok", "message": "✅ OTP sent to your email. Enter OTP to verify."}

@app.post("/verify-register-otp")
async def verify_register_otp(username: str = Form(...), otp: str = Form(...)):
    cache = _load_otp_cache()
    entry = cache.get(username)

    if not entry:
        return JSONResponse({"status": "error", "message": "No OTP session found"}, status_code=400)

    if entry.get("context") != "register":
        return JSONResponse({"status": "error", "message": "OTP context mismatch"}, status_code=400)

    if entry.get("otp") != otp:
        return JSONResponse({"status": "error", "message": "Incorrect OTP"}, status_code=400)

    if datetime.utcnow() > datetime.fromisoformat(entry.get("expiry")):
        del cache[username]
        _save_otp_cache(cache)
        return JSONResponse({"status": "error", "message": "OTP expired"}, status_code=400)

    pending = entry.get("pending_user")
    if not pending:
        return JSONResponse({"status": "error", "message": "No pending user data"}, status_code=400)

    users = safe_load_json(USERS_FILE, {})
    users[username] = pending
    safe_save_json(USERS_FILE, users)

    del cache[username]
    _save_otp_cache(cache)

    return {"status": "ok", "message": "✅ Email Verified & Account Created Successfully!"}


@app.post("/login")
async def login(username: str = Form(...), password: str = Form(...)):
    users = safe_load_json(USERS_FILE, {})
    if username not in users:
        return JSONResponse({"status":"error","message":"Invalid credentials"}, status_code=401)
    if users[username].get("password_hash") != sha_str(password):
        return JSONResponse({"status":"error","message":"Invalid credentials"}, status_code=401)
    email = users[username].get("email")
    if not email:
        return JSONResponse({"status":"error","message":"User has no email set"}, status_code=400)
    ok, otp = create_and_send_otp(username, email, context="login")
    if not ok:
        return JSONResponse({"status":"error","message":"Failed to send OTP"}, status_code=500)
    return {"status":"ok","message":"OTP sent to registered email"}

@app.post("/verify-login-otp")
async def verify_login_otp(username: str = Form(...), otp: str = Form(...)):
    ok, msg = verify_otp(username, otp, context="login")
    if not ok:
        return JSONResponse({"status": "error", "message": msg}, status_code=400)

    users = safe_load_json(USERS_FILE, {})
    role = users.get(username, {}).get("role", "user")

    # ✅ Return role cleanly & clearly
    return {
        "status": "ok",
        "role": role
    }


# File upload/download/classification endpoints - same flow as before, with encryption & classification
@app.post("/upload")
async def upload(username: str = Form(...), file: UploadFile = File(...)):
    users = safe_load_json(USERS_FILE, {})
    if username not in users:
        return JSONResponse({"status":"error","message":"Unknown user"}, status_code=400)

    raw = await file.read()
    file_hash = sha_bytes(raw)
    chain = load_chain()
    for e in chain:
        if e.get("hash") == file_hash:
            return JSONResponse({"status":"error","message":"Duplicate file detected"}, status_code=409)

    text = extract_text_from_pdf_bytes(raw)
    debug_info = {"ocr_length": len(text), "ocr_snippet": text[:500]}

    doc_type = "unknown"
    confidence = 0.0
    source = "none"
    explanation = ""
    ollama_meta = None

    # Try Ollama
    try:
        label, conf, raw_resp = await classify_document_llm_ollama(text) if text else (None, None, None)
        debug_info["ollama_label"] = label
        debug_info["ollama_conf"] = conf
        debug_info["ollama_raw"] = raw_resp
        if label:
            doc_type = label
            confidence = conf if conf is not None else 0.0
            source = "llama"
            explanation = raw_resp if isinstance(raw_resp, str) else ""
            ollama_meta = raw_resp
    except Exception as e:
        debug_info["ollama_error"] = str(e)

    # Fallback to MiniLM
    if (doc_type == "unknown" or confidence < 0.2):
        if text and MINI_MODEL:
            try:
                label, score = classify_minilm(text)
                debug_info["minilm"] = {"label": label, "score": score}
                if score >= 0.2:
                    doc_type = label
                    confidence = score
                    source = "minilm"
                    explanation = explanation or f"MiniLM score {score:.3f}"
            except Exception as e:
                debug_info["minilm_error"] = str(e)

    # Encrypt & Save file
    user_dir = os.path.join(USER_DATA, username)
    os.makedirs(user_dir, exist_ok=True)
    encrypted_filename = file.filename + ".enc"
    target_path = os.path.join(user_dir, encrypted_filename)
    try:
        enc_b64 = encrypt_bytes(raw)
        with open(target_path, "w", encoding="utf-8") as f:
            f.write(enc_b64)
    except Exception as e:
        debug_info["save_error"] = str(e)
        save_debug({"stage":"upload_save_error","debug": debug_info})
        return JSONResponse({"status":"error","message":"Failed to save file"}, status_code=500)

    entry = {
        "user": username,
        "filename": encrypted_filename,
        "orig_filename": file.filename,
        "hash": file_hash,
        "type": doc_type,
        "type_confidence": float(confidence),
        "type_source": source,
        "type_explanation": explanation,
        "time": str(datetime.now())
    }
    chain.append(entry)
    save_chain(chain)

    logs = load_logs()
    logs.append({"user": username, "file": encrypted_filename, "time": str(datetime.now()), "doc_type": doc_type})
    save_logs(logs)
    save_debug({"stage":"upload_complete","debug": debug_info, "entry": entry, "ollama_meta": ollama_meta})

    return {"status":"success","doc_type": doc_type, "confidence": float(confidence), "source": source, "hash": file_hash, "explanation": explanation, "debug": debug_info}

@app.get("/download/{username}/{filename}")
async def download_file(username: str, filename: str):
    p = os.path.join(USER_DATA, username, filename)
    if not os.path.exists(p):
        return JSONResponse({"status":"error","message":"Not found"}, status_code=404)
    try:
        with open(p, "r", encoding="utf-8") as f:
            enc_b64 = f.read()
        dec_bytes = decrypt_bytes(enc_b64)
        tmp_name = os.path.join(BASE, "data", f"tmp_{username}_{filename.replace('/','_')}")
        with open(tmp_name, "wb") as t:
            t.write(dec_bytes)
        chain = load_chain()
        entry = next((e for e in chain if e.get("filename")==filename and e.get("user")==username), None)
        download_name = entry.get("orig_filename") if entry and entry.get("orig_filename") else filename.replace(".enc","")
        return FileResponse(tmp_name, filename=download_name)
    except Exception as e:
        return JSONResponse({"status":"error","message": f"Decryption failed: {e}"}, status_code=500)

@app.get("/user/files/{username}")
async def get_user_files(username: str):
    folder = os.path.join(USER_DATA, username)
    if not os.path.exists(folder):
        return []
    files = sorted(os.listdir(folder))
    chain = load_chain()
    out = []
    for fname in files:
        meta = next((e for e in chain if e.get("user")==username and e.get("filename")==fname), {})
        out.append({
            "filename": fname,
            "orig_filename": meta.get("orig_filename", fname.replace(".enc","")),
            "type": meta.get("type", "unknown"),
            "confidence": meta.get("type_confidence", 0.0),
            "source": meta.get("type_source", ""),
            "time": meta.get("time", "")
        })
    return out

@app.get("/admin/files")
async def admin_files():
    return load_chain()

@app.get("/admin/logs")
async def admin_logs():
    return load_logs()

@app.post("/admin/generate_master_key")
async def admin_generate_master_key(username: str = Form(...)):
    users = safe_load_json(USERS_FILE, {})
    if username not in users or users[username].get("role") != "admin":
        return JSONResponse({"status":"error","message":"Not authorized"}, status_code=401)
    try:
        generate_and_store_master_key(MASTER_KEY_PATH)
        return {"status":"success","message":"Master key generated"}
    except Exception as e:
        return JSONResponse({"status":"error","message": str(e)}, status_code=500)

@app.get("/debug/last")
async def debug_last():
    return safe_load_json(DEBUG_FILE, {})

@app.get("/admin/export")
async def admin_export():
    chain = load_chain()
    logs = load_logs()
    if REPORTLAB_AVAILABLE:
        try:
            tmp = os.path.join(BASE, "system_report.pdf")
            c = canvas.Canvas(tmp, pagesize=letter)
            width, height = letter
            y = height - 40
            c.setFont("Helvetica-Bold", 14)
            c.drawString(30, y, "DocSec System Report")
            y -= 30
            c.setFont("Helvetica", 9)
            c.drawString(30, y, f"Generated: {datetime.now()}")
            y -= 20
            for e in chain:
                line = f"{e.get('user')} | {e.get('orig_filename','?')} | {e.get('type')} | {e.get('hash')[:12]}... | {e.get('time')}"
                c.drawString(30, y, line)
                y -= 12
                if y < 60:
                    c.showPage(); y = height - 40
            c.showPage()
            y = height - 40
            c.setFont("Helvetica-Bold", 12)
            c.drawString(30, y, "Login Logs:")
            y -= 16
            c.setFont("Helvetica", 9)
            for l in logs:
                line = f"{l.get('user')} | {l.get('time')}"
                c.drawString(30, y, line)
                y -= 12
                if y < 60:
                    c.showPage(); y = height - 40
            c.save()
            return FileResponse(tmp, filename="DocSec_System_Report.pdf")
        except Exception as e:
            print("[WARN] export failed:", e)
    tmp = os.path.join(BASE, "system_report.json")
    safe_save_json(tmp, {"generated": str(datetime.now()), "chain": chain, "logs": logs})
    return FileResponse(tmp, filename="system_report.json")

@app.get("/health")
async def health():
    users = safe_load_json(USERS_FILE, {})
    return {"status":"ok", "users": len(users)}

# ---------------- helper functions used above (moved here to avoid reference errors) ----------------
def load_chain():
    return safe_load_json(CHAIN_FILE, [])

def save_chain(chain):
    safe_save_json(CHAIN_FILE, chain)

def load_logs():
    return safe_load_json(LOGIN_LOG, [])

def save_logs(logs):
    safe_save_json(LOGIN_LOG, logs)
