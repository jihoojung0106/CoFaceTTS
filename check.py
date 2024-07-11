import torch

# 모델 파일 로드
model_path = 'facetts_lrs3.pt'
model = torch.load(model_path)

# 모델 구조 출력
print(model)

# 모델의 키워드(파라미터 이름) 출력
for name, param in model.named_parameters():
    print(name)
