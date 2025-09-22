from dataclasses import dataclass
from typing import List, Dict, Optional

import streamlit as st
import google.generativeai as genai

from dotenv import load_dotenv
import os

load_dotenv()  # otomatis baca file .env

api_key = os.getenv("GOOGLE_API_KEY")
	


@dataclass
class GeminiLLM:
	model: any

	def generate(self, prompt: str, temperature: float = 0.7, max_new_tokens: int = 512) -> str:
		resp = self.model.generate_content(
			prompt,
			generation_config={
				"temperature": float(temperature),
				"max_output_tokens": int(max_new_tokens),
			}
		)
		return (getattr(resp, "text", None) or "").strip()


@st.cache_resource(show_spinner=False)
def load_llm(model_name: str, api_key: Optional[str] = None) -> GeminiLLM:
	if not api_key:
		raise RuntimeError("GOOGLE_API_KEY tidak ditemukan. Isi di sidebar atau set di environment.")
	genai.configure(api_key=api_key)
	model = genai.GenerativeModel(model_name or "gemini-1.5-flash")
	return GeminiLLM(model=model)


def compose_prompt(
	messages: List[Dict[str, str]],
	domain: str,
	gaya: str,
	knowledge_snippets: List[str],
	max_history: int = 3,
) -> str:
	style = "formal" if gaya.lower() == "formal" else "santai"
	context = "\n".join([f"- {s}" for s in knowledge_snippets]) if knowledge_snippets else "(tidak ada konteks khusus)"

	# ambil history terakhir
	recent = messages[-(max_history*2):] if max_history > 0 else messages
	history_text = "\n".join([f"{m['role']}: {m['content']}" for m in recent])

	base = f"Anda adalah asisten AI berbahasa Indonesia dengan gaya {style}. "
	if domain and domain.strip():
		system = base + f"Fokus domain: {domain}. Gunakan konteks jika relevan, dan jawab ringkas, jelas, dan akurat."
	else:
		system = base + "Gunakan konteks jika relevan, dan jawab ringkas, jelas, dan akurat."

	prompt = (
		f"[SYSTEM]\n{system}\n\n"
		f"[KONTEXT]\n{context}\n\n"
		f"[RIWAYAT]\n{history_text}\n\n"
		f"[TUGAS]\nBalas pesan pengguna terakhir secara {style}. Jika pertanyaan di luar domain, jawab secara umum namun tetap bermanfaat."
	)
	return prompt


def postprocess_response(text: str, gaya: str, domain: str) -> str:
	text = text.strip()
	if gaya.lower() == "formal":
		# kecilkan emoji/ekspresi santai
		text = text.replace("😀", "").replace("😊", "").replace("😅", "")
		# kapitalisasi kalimat pertama secara sederhana
		if text and text[0].islower():
			text = text[0].upper() + text[1:]
	return text


def suggest_next_actions(user_text: str, domain: str) -> List[str]:
	lower = user_text.lower()
	ideas: List[str] = []
	if domain.lower() == "edukasi":
		if any(k in lower for k in ["materi", "belajar", "ringkas"]):
			ideas.append("Minta rangkuman poin-poin kunci dari topik tertentu.")
		ideas.append("Minta contoh soal dan pembahasannya.")
		ideas.append("Minta rencana belajar mingguan.")
	elif domain.lower() == "gizi":
		ideas.append("Minta estimasi kebutuhan kalori harian.")
		ideas.append("Minta contoh menu seimbang 1 hari.")
		ideas.append("Tanyakan alternatif pengganti bahan makanan tertentu.")
	elif domain.lower() == "travel":
		ideas.append("Minta itinerary singkat untuk 3 hari.")
		ideas.append("Tanyakan estimasi biaya perjalanan.")
		ideas.append("Minta tips transportasi lokal.")
	else:  # Produktivitas
		ideas.append("Minta to-do list prioritas harian.")
		ideas.append("Minta template pomodoro untuk 2 jam kerja.")
		ideas.append("Minta ringkasan notulen rapat.")
	return ideas[:3]



