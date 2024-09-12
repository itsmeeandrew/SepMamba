# SepMamba: State-space models for speaker separation using Mamba

<p align=center><em>
Thor Højhus Avenstrup, Boldizsár Elek, István László Mádi, András Bence Schin,<br />
Morten Mørup, Bjørn Sand Jensen, Kenny Olsen <br />
Technical University of Denmark (DTU)
</em></p>

**Deep learning-based single-channel speaker separa- tion has improved significantly in recent years in large part due to the introduction of the transformer-based attention mechanism. However, these improvements come with intense computational demands, precluding their use in many practical applications. As a computationally efficient alternative with similar modeling capabilities, Mamba was recently introduced. We propose Sep- Mamba, a U-Net-based architecture composed of bidirectional Mamba layers. We find that our approach outperforms similarly- sized prominent models — including transformer-based models — on the WSJ0 2-speaker dataset while enjoying significant computational benefits in terms of multiply-accumulates, peak memory usage, and wall-clock time. We additionally report strong results for causal variants of SepMamba. Our approach provides a computationally favorable alternative to transformer- based architectures for deep speech separation.**

![network](https://github.com/user-attachments/assets/3f8897ee-0297-4464-901c-12befc0a1a46)
