import torch

def get_net(weights_path, device='cpu'):
    """
    YOLOPv2 모델을 로드합니다.
    
    Args:
    weights_path (str): 모델 가중치 파일의 경로
    device (str): 사용할 디바이스 ('cpu' 또는 'cuda')
    
    Returns:
    torch.nn.Module: 로드된 YOLOPv2 모델
    """
    model = torch.jit.load(weights_path, map_location=device)
    model.eval()
    return model

# 이 줄을 추가하여 get_net 함수를 명시적으로 export합니다
__all__ = ['get_net']