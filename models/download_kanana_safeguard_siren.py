import os
from transformers import AutoTokenizer, AutoModelForCausalLM

def download_kanana_siren(model_id: str, save_path: str):
    os.makedirs(save_path, exist_ok=True)

    print(f"🔽 토크나이저 다운로드 중... ({model_id})")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.save_pretrained(save_path)
    print(f"✅ 토크나이저 저장 완료: {save_path}")

    print(f"🔽 모델 다운로드 중... ({model_id})")
    model = AutoModelForCausalLM.from_pretrained(model_id)
    model.save_pretrained(save_path)
    print(f"✅ 모델 저장 완료: {save_path}")

if __name__ == "__main__":
    model_id = "kakaocorp/kanana-safeguard-siren-8b"
    save_path = "/home/lsw9128/sfmi-guardrail/models/kakaocorp/kanana-safeguard-siren-8b"

    print(f"📦 모델을 {save_path} 경로에 저장합니다...\n")
    download_kanana_siren(model_id, save_path)
    print("\n🎉 모든 파일 저장이 완료되었습니다.")
