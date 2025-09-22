import streamlit as st
from retrieval import KnowledgeBase
from utils import load_llm, compose_prompt, postprocess_response, suggest_next_actions


import os

st.set_page_config(page_title="AI Chatbot - Edukasi/Gizi/Travel/Produktivitas", page_icon="🤖", layout="centered")

	
def init_session_state() -> None:
	if "history" not in st.session_state:
		st.session_state.history = []  # list of dicts: {"role": "user"|"assistant", "content": str}
	if "kb" not in st.session_state:
		st.session_state.kb = None
	if "model_name" not in st.session_state:
		st.session_state.model_name = "gemini-1.5-flash"


def sidebar_controls() -> dict:
	st.sidebar.header("Pengaturan Chatbot")
	gaya = st.sidebar.radio("Gaya Bahasa", ["Formal", "Santai"], index=0)
	model_name = st.sidebar.text_input("Nama Model Gemini", "gemini-1.5-flash", help="Contoh: gemini-1.5-flash atau gemini-1.5-pro")
	api_key_override = st.sidebar.text_input("Google API Key", value="", type="password", help="Masukkan kunci API dari Google AI Studio.")
	col1, col2 = st.sidebar.columns(2)
	with col1:
		clear_chat = st.button("Clear Chat")
	with col2:
		new_chat = st.button("Chat Baru")
	st.sidebar.caption(
		"Catatan: Gunakan API Key dari Google AI Studio (Generative Language API)."
	)
	return {
		"gaya": gaya,
		"model_name": model_name,
		"api_key_override": api_key_override,
		"clear_chat": clear_chat,
		"new_chat": new_chat,
	}


def ensure_kb(domain: str) -> KnowledgeBase:
	if (
		st.session_state.kb is None
		or st.session_state.kb.domain.lower() != domain.lower()
	):
		st.session_state.kb = KnowledgeBase(domain=domain)
	return st.session_state.kb


def render_chat():
	st.title("🤖 Chatbot Pintar")
	st.caption("Chatbot berbasis Google Gemini. Masukkan API key di sidebar, pilih model, lalu mulai bertanya.")

	controls = sidebar_controls()
	# Domain kosong: tidak menampilkan label domain pada prompt
	domain_const = ""
	# Aksi kontrol chat
	if controls.get("clear_chat") or controls.get("new_chat"):
		st.session_state.history = []
		st.rerun()
	kb = ensure_kb(domain_const)
	st.session_state.model_name = controls["model_name"]
	api_key = controls.get("api_key_override")
	llm = load_llm(controls["model_name"], api_key)

	# Tampilkan riwayat percakapan
	for msg in st.session_state.history:
		with st.chat_message(msg["role"]):
			st.markdown(msg["content"])

	# Input pengguna
	user_input = st.chat_input("Ketik pertanyaan atau perintah Anda...")
	if user_input:
		st.session_state.history.append({"role": "user", "content": user_input})

		# Ambil konteks dari KB
		kb_results = kb.search(user_input, k=4)

		# Komposisi prompt
		prompt = compose_prompt(
			messages=st.session_state.history,
			domain=domain_const,
			gaya=controls["gaya"],
			knowledge_snippets=kb_results,
		)

		# Generate
		with st.spinner("Menyusun jawaban..."):
			raw_response = llm.generate(prompt, temperature=0.7, max_new_tokens=256)
			assistant_text = postprocess_response(
				raw_response,
				gaya=controls["gaya"],
				domain=domain_const,
			)

		st.session_state.history.append({"role": "assistant", "content": assistant_text})
		with st.chat_message("assistant"):
			st.markdown(assistant_text)

		# Rekomendasi tindak lanjut (berbasis intent sederhana)
		with st.expander("Rekomendasi tindak lanjut"):
			for s in suggest_next_actions(user_input, domain_const):
				st.markdown(f"- {s}")


def main():
	init_session_state()
	render_chat()


if __name__ == "__main__":
	main()



