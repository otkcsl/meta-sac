meta-sac は、本研究で提案する新たな学習手法を実装したリポジトリです。本手法では、探索を重視する方針と活用を重視する方針を持つ2種類のAIを同時に学習させ、両者が協調することで学習を進める枠組みを導入しています。これにより、従来手法で必要とされていた温度パラメータの個別調整を行うことなく、安定した性能向上を実現することを目指しています。

pip install PyYAML==6.0.3
pip install pandas==1.5.0
pip install "gymnasium[mujoco]"==0.29.0
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
