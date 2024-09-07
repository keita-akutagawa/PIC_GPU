Particle in Cellシミュレーションのコードです。\
C++とCUDAで書かれています。

CUDAとThrustライブラリを用いてGPU並列化を施しています。\
MPIによってマルチGPU並列化を施しています。

【MEMO】\
今はマルチGPU並列化に取り組んでいます。

## スキーム
- Yee lattice
- Langdon-Marder type correction
