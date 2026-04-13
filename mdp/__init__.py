__version__ = "0.1.0"

# ── 서드파티 노이즈 억제 ──
# transformers, datasets, torch 등이 뿜는 FutureWarning과 verbose 로그를 필터링한다.
# mdp 자체 로그(logger = getLogger(__name__))는 영향받지 않는다.

import logging as _logging
import warnings as _warnings

# FutureWarning: 의존성 라이브러리의 API 변경 예고. 사용자가 조치할 수 없음.
_warnings.filterwarnings("ignore", category=FutureWarning)
# torch의 beta feature UserWarning (e.g. torch.compile, cuDNN)
_warnings.filterwarnings("ignore", message=".*beta.*", module="torch")

# 의존성 로거: INFO가 매우 장황하므로 WARNING 이상만 출력
for _name in ("transformers", "datasets", "accelerate", "bitsandbytes"):
    _logging.getLogger(_name).setLevel(_logging.WARNING)
