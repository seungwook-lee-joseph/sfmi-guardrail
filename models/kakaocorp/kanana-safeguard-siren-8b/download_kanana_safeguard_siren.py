import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification


def download_kanana_model(model_id: str, save_path: str):
    os.makedirs(save_path, exist_ok=True)

    print(f"🔽 토크나이저 다운로드 중... ({model_id})")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.save_pretrained(save_path)
    print(f"✅ 토크나이저 저장 완료: {save_path}")

    print(f"🔽 모델 다운로드 중... ({model_id})")
    model = AutoModelForSequenceClassification.from_pretrained(model_id)
    model.save_pretrained(save_path)
    print(f"✅ 모델 저장 완료: {save_path}")


if __name__ == "__main__":
    # 모델 ID (Hugging Face 기준 이름)
    model_id = "kakaocorp/kanana-safeguard-8b/kanana-safeguard-siren-8b"
    # 저장 경로
    save_path = os.path.join(os.getcwd(), "models/kanana-safeguard-siren-8b")

    print(f"📦 모델을 {save_path} 경로에 저장합니다...\n")
    download_kanana_model(model_id, save_path)
    print("\n🎉 모든 파일 저장이 완료되었습니다.")
